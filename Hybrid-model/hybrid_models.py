import json
import torch
import torch.nn as nn
from transformers import GPT2Config, GPT2Model, DistilBertModel, GPT2TokenizerFast, DistilBertTokenizerFast, GPT2Tokenizer, GPT2LMHeadModel
import os
import time
import re
import gc
import psutil
from flask import Flask, request, jsonify
from threading import Lock
from typing import Dict, Optional

app = Flask(__name__)

def log_to_file(message):
    print(message)

# Concatenation Fusion Layer
class FusionLayer(nn.Module):
    def __init__(self, gpt2_dim: int, bert_dim: int, output_dim: int, dropout_rate: float):
        super().__init__()
        self.gpt2_proj = nn.Linear(gpt2_dim, output_dim)
        self.bert_proj = nn.Linear(bert_dim, output_dim)
        self.fusion = nn.Sequential(
            nn.Linear(output_dim * 2, output_dim),
            nn.Tanh(),
            nn.Dropout(dropout_rate)
        )
        self.layer_norm = nn.LayerNorm(output_dim)

    def forward(self, gpt2_features: torch.Tensor, bert_features: torch.Tensor,
                attention_mask: torch.Tensor) -> torch.Tensor:
        gpt2_proj = self.gpt2_proj(gpt2_features)
        bert_proj = self.bert_proj(bert_features)
        concat_features = torch.cat([gpt2_proj, bert_proj], dim=-1)
        fused = self.fusion(concat_features)
        return self.layer_norm(fused)

# Cross-Attention Fusion Layer
class CrossAttentionFusionLayer(nn.Module):
    def __init__(self, gpt2_dim: int, bert_dim: int, output_dim: int, dropout_rate: float, num_heads: int = 8):
        super().__init__()
        self.gpt2_proj = nn.Linear(gpt2_dim, output_dim)
        self.bert_proj = nn.Linear(bert_dim, output_dim)
        self.cross_attention = nn.MultiheadAttention(embed_dim=output_dim, num_heads=num_heads, dropout=dropout_rate, batch_first=True)
        self.dropout = nn.Dropout(dropout_rate)
        self.layer_norm = nn.LayerNorm(output_dim)

    def forward(self, gpt2_features: torch.Tensor, bert_features: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        gpt2_proj = self.gpt2_proj(gpt2_features)
        bert_proj = self.bert_proj(bert_features)
        attn_mask = attention_mask.float().masked_fill(attention_mask == 0, float('-inf')).masked_fill(attention_mask == 1, 0)
        fused_features, _ = self.cross_attention(
            query=gpt2_proj,
            key=bert_proj,
            value=bert_proj,
            key_padding_mask=attention_mask == 0
        )
        fused_features = self.dropout(fused_features) + gpt2_proj
        return self.layer_norm(fused_features)

# Dense Fusion Layer
class DenseFusionLayer(nn.Module):
    def __init__(self, gpt2_dim: int, bert_dim: int, output_dim: int, dropout_rate: float):
        super().__init__()
        self.gpt2_proj = nn.Linear(gpt2_dim, output_dim)
        self.bert_proj = nn.Linear(bert_dim, output_dim)
        self.dense = nn.Linear(output_dim, output_dim)
        self.activation = nn.Tanh()
        self.dropout = nn.Dropout(dropout_rate)
        self.layer_norm = nn.LayerNorm(output_dim)

    def forward(self, gpt2_features: torch.Tensor, bert_features: torch.Tensor,
                attention_mask: torch.Tensor) -> torch.Tensor:
        gpt2_proj = self.gpt2_proj(gpt2_features)
        bert_proj = self.bert_proj(bert_features)
        combined_features = gpt2_proj + bert_proj
        fused_features = self.dense(combined_features)
        fused_features = self.activation(fused_features)
        fused_features = self.dropout(fused_features)
        return self.layer_norm(fused_features)

# Unified Hybrid Model with Fusion Type Selection
class HybridGPT2DistilBERTMultiTask(nn.Module):
    def __init__(self, num_intents: int, num_categories: int, num_ner_labels: int,
                 dropout_rate: float, fusion_type: str = "concat",
                 loss_weights: Optional[Dict[str, float]] = None,
                 ner_class_weights: Optional[torch.Tensor] = None,
                 category_class_weights: Optional[torch.Tensor] = None,
                 intent_class_weights: Optional[torch.Tensor] = None):
        super().__init__()
        log_to_file(f"Initializing model with {fusion_type} fusion...")
        self.gpt2_config = GPT2Config.from_pretrained('gpt2')
        self.gpt2 = GPT2Model.from_pretrained('gpt2')
        self.distilbert = DistilBertModel.from_pretrained('distilbert-base-uncased')

        for param in self.gpt2.parameters():
            param.requires_grad = False
        for param in self.distilbert.parameters():
            param.requires_grad = False
        log_to_file("All GPT-2 and DistilBERT layers remain frozen")

        gpt2_dim = self.gpt2_config.n_embd
        bert_dim = self.distilbert.config.hidden_size
        hidden_size = gpt2_dim

        if fusion_type == "concat":
            self.fusion_layer = FusionLayer(gpt2_dim, bert_dim, hidden_size, dropout_rate)
        elif fusion_type == "crossattention":
            self.fusion_layer = CrossAttentionFusionLayer(gpt2_dim, bert_dim, hidden_size, dropout_rate)
        elif fusion_type == "dense":
            self.fusion_layer = DenseFusionLayer(gpt2_dim, bert_dim, hidden_size, dropout_rate)
        else:
            raise ValueError("fusion_type must be 'concat', 'crossattention', or 'dense'")

        self.intent_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size), nn.Tanh(), nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, num_intents)
        )
        self.category_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size), nn.Tanh(), nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, num_categories)
        )
        self.ner_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size), nn.Tanh(), nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, num_ner_labels)
        )

        self.loss_weights = loss_weights or {'intent': 0.3, 'category': 0.3, 'ner': 0.4}
        self.intent_loss_fn = nn.CrossEntropyLoss(weight=intent_class_weights) if intent_class_weights is not None else nn.CrossEntropyLoss()
        self.category_loss_fn = nn.CrossEntropyLoss(weight=category_class_weights) if category_class_weights is not None else nn.CrossEntropyLoss()
        self.ner_loss_fn = nn.CrossEntropyLoss(weight=ner_class_weights) if ner_class_weights is not None else nn.CrossEntropyLoss()

    def forward(self, gpt2_input_ids, gpt2_attention_mask,
                distilbert_input_ids, distilbert_attention_mask,
                intent_labels=None, category_labels=None, ner_labels=None):
        gpt2_outputs = self.gpt2(input_ids=gpt2_input_ids, attention_mask=gpt2_attention_mask)
        distilbert_outputs = self.distilbert(input_ids=distilbert_input_ids, attention_mask=distilbert_attention_mask)

        gpt2_features = gpt2_outputs.last_hidden_state
        bert_features = distilbert_outputs.last_hidden_state

        fused_features = self.fusion_layer(gpt2_features, bert_features, gpt2_attention_mask)

        masked_features = fused_features * gpt2_attention_mask.unsqueeze(-1)
        sequence_repr = masked_features.sum(dim=1) / gpt2_attention_mask.sum(dim=1, keepdim=True)

        intent_logits = self.intent_head(sequence_repr)
        category_logits = self.category_head(sequence_repr)
        ner_logits = self.ner_head(fused_features)

        output_dict = {
            'intent_logits': intent_logits,
            'category_logits': category_logits,
            'ner_logits': ner_logits
        }

        if all(label is not None for label in [intent_labels, category_labels, ner_labels]):
            intent_loss = self.intent_loss_fn(intent_logits, intent_labels)
            category_loss = self.category_loss_fn(category_logits, category_labels)
            combined_mask = (gpt2_attention_mask * distilbert_attention_mask)
            active_loss = combined_mask.view(-1) == 1
            active_logits = ner_logits.view(-1, ner_logits.size(-1))[active_loss]
            active_labels = ner_labels.view(-1)[active_loss]
            ner_loss = self.ner_loss_fn(active_logits, active_labels)

            total_loss = (self.loss_weights['intent'] * intent_loss +
                          self.loss_weights['category'] * category_loss +
                          self.loss_weights['ner'] * ner_loss)

            output_dict.update({
                'loss': total_loss,
                'intent_loss': intent_loss,
                'category_loss': category_loss,
                'ner_loss': ner_loss
            })

        return output_dict

def inference_hybrid(model, text, gpt2_tokenizer, distilbert_tokenizer, label_encoders, max_length, device):
    model.eval()
    gpt2_encoding = gpt2_tokenizer(text, max_length=max_length, padding="max_length", truncation=True, return_tensors="pt")
    distilbert_encoding = distilbert_tokenizer(text, max_length=max_length, padding="max_length", truncation=True, return_tensors="pt")

    inputs = {
        "gpt2_input_ids": gpt2_encoding["input_ids"].to(device),
        "gpt2_attention_mask": gpt2_encoding["attention_mask"].to(device),
        "distilbert_input_ids": distilbert_encoding["input_ids"].to(device),
        "distilbert_attention_mask": distilbert_encoding["attention_mask"].to(device)
    }

    with torch.no_grad():
        outputs = model(**inputs)

    intent_logits = outputs["intent_logits"]
    category_logits = outputs["category_logits"]
    ner_logits = outputs["ner_logits"]

    intent_probs = torch.nn.functional.softmax(intent_logits, dim=-1)[0]
    category_probs = torch.nn.functional.softmax(category_logits, dim=-1)[0]
    ner_probs = torch.nn.functional.softmax(ner_logits, dim=-1)

    intent_pred = torch.argmax(intent_probs).cpu().item()
    intent_confidence = intent_probs[intent_pred].cpu().item()
    category_pred = torch.argmax(category_probs).cpu().item()
    category_confidence = category_probs[category_pred].cpu().item()
    ner_preds = torch.argmax(ner_probs, dim=-1).cpu().numpy()[0]
    ner_confidences = torch.max(ner_probs, dim=-1)[0][0].cpu().numpy()

    intent_decoder = {v: k for k, v in label_encoders["intent_encoder"].items()}
    category_decoder = {v: k for k, v in label_encoders["category_encoder"].items()}
    ner_decoder = {v: k for k, v in label_encoders["ner_label_encoder"].items()}

    intent_label = intent_decoder[intent_pred]
    category_label = category_decoder[category_pred]
    tokens = gpt2_tokenizer.convert_ids_to_tokens(inputs["gpt2_input_ids"][0].tolist())
    seq_len = int(inputs["gpt2_attention_mask"][0].sum().item())
    ner_labels = [ner_decoder[pred] for pred in ner_preds[:seq_len]]

    entities = []
    current_entity = None
    entity_tokens = []
    entity_confidences = []
    entity_type = None

    for i, (token, label, confidence) in enumerate(zip(tokens[:seq_len], ner_labels, ner_confidences[:seq_len])):
        if label.startswith("B-"):
            if current_entity is not None:
                entity_text = gpt2_tokenizer.convert_tokens_to_string(entity_tokens).strip()
                if entity_text:
                    avg_confidence = sum(entity_confidences) / len(entity_confidences)
                    entities.append({"entity": entity_text, "label": entity_type, "confidence": avg_confidence})
            current_entity = label[2:]
            entity_type = label[2:]
            entity_tokens = [token]
            entity_confidences = [confidence]
        elif label.startswith("I-") and current_entity == label[2:]:
            entity_tokens.append(token)
            entity_confidences.append(confidence)
        elif current_entity is not None:
            entity_text = gpt2_tokenizer.convert_tokens_to_string(entity_tokens).strip()
            if entity_text:
                avg_confidence = sum(entity_confidences) / len(entity_confidences)
                entities.append({"entity": entity_text, "label": entity_type, "confidence": avg_confidence})
            current_entity = None
            entity_tokens = []
            entity_confidences = []
            entity_type = None

    if current_entity is not None:
        entity_text = gpt2_tokenizer.convert_tokens_to_string(entity_tokens).strip()
        if entity_text:
            avg_confidence = sum(entity_confidences) / len(entity_confidences)
            entities.append({"entity": entity_text, "label": entity_type, "confidence": avg_confidence})

    return {
        "intent": {"label": intent_label, "confidence": intent_confidence},
        "category": {"label": category_label, "confidence": category_confidence},
        "ner": entities
    }

def generate_response(model, tokenizer, instruction, classification, history, max_length=512, device="cuda"):
    model.eval()
    intent = classification["intent"]["label"] if isinstance(classification["intent"], dict) else classification["intent"]
    category = classification["category"]["label"] if isinstance(classification["category"], dict) else classification["category"]
    if isinstance(intent, str) and "[" in intent:
        intent = intent.strip("[]'")
    if isinstance(category, str) and "[" in category:
        category = category.strip("[]'")
    entities_text = ", ".join([f"{entity['entity']} ({entity['label']})" for entity in classification["ner"]]) if classification["ner"] else "none"

    history_text = ""
    if history:
        history_text = "Previous conversation:\n" + "\n".join([f"User: {h['instruction']}\nAssistant: {h['response']}" for h in history]) + "\n\n"

    input_text = f"[INST] {history_text}Current query: {instruction}\n\nBased on the following classification:\n- Intent: {intent}\n- Category: {category}\n- Entities: {entities_text}\n\nProvide a helpful customer service response: [RESP]"

    input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)
    attention_mask = torch.ones_like(input_ids).to(device)

    try:
        with torch.no_grad():
            output_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=max_length,
                num_beams=5,
                no_repeat_ngram_size=2,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=False)
        if "[RESP]" in generated_text:
            response = generated_text.split("[RESP]")[1].strip()
            if "[EOS]" in response:
                response = response.split("[EOS]")[0].strip()
        else:
            response = generated_text[len(input_text):].strip()
        steps_pattern = re.search(r'(\d+)\.\s+([A-Z])', response)
        if steps_pattern or "step" in response.lower() or "follow" in response.lower():
            for i in range(1, 10):
                step_marker = f"{i}. "
                if step_marker in response and f"\n{i}. " not in response:
                    response = response.replace(step_marker, f"\n{i}. ")
            response = re.sub(r'\n\s*\n', '\n\n', response)
            response = response.lstrip('\n')
        response = re.sub(r'https?://\S+', '', response)
        response = re.sub(r'<[^>]*>', '', response)
        response = re.sub(r'\{\s*"[^"]*":', '', response)
        response = re.sub(r'\s+', ' ', response).strip()
        return response
    except Exception as e:
        print(f"Error in generate_response: {e}")
        return f"I apologize, but I couldn't generate a response. Error: {str(e)}"

def get_memory_usage():
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return mem_info.rss / 1024 / 1024

def get_peak_memory_usage(func, *args, **kwargs):
    device_param = kwargs.pop('device', None)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    start_mem = get_memory_usage()
    result = func(*args, **kwargs)
    end_mem = get_memory_usage()
    peak_gpu_mem = 0
    if torch.cuda.is_available() and device_param == "cuda":
        peak_gpu_mem = torch.cuda.max_memory_allocated() / 1024 / 1024
        return result, peak_gpu_mem
    else:
        return result, max(0, end_mem - start_mem)

class ConversationalInferenceService:
    def __init__(self, model_paths, fusion_type="concat"):
        self.fusion_type = fusion_type
        self.output_dir = model_paths[f"{fusion_type}_model_dir"]
        self.generation_model_path = model_paths["generation_model_path"]
        self.generation_tokenizer_path = model_paths["generation_tokenizer_path"]
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.conversation_history = {}
        self.max_history = 5
        self.lock = Lock()
        self.load_models()

    def load_models(self):
        print(f"\nLoading {self.fusion_type} model on {self.device.upper()}...")
        encoders_path = os.path.join(self.output_dir, "label_encoders.json")
        hyperparams_path = os.path.join(self.output_dir, "hyperparameters.json")
        model_path = os.path.join(self.output_dir, "hybrid_model.pth")

        with open(encoders_path, 'r', encoding='utf-8') as f:
            self.label_encoders = json.load(f)
        with open(hyperparams_path, 'r', encoding='utf-8') as f:
            self.hyperparameters = json.load(f)

        self.gpt2_tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
        self.distilbert_tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
        if self.gpt2_tokenizer.pad_token is None:
            self.gpt2_tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        if self.distilbert_tokenizer.pad_token is None:
            self.distilbert_tokenizer.add_special_tokens({'pad_token': '[PAD]'})

        self.classification_model = HybridGPT2DistilBERTMultiTask(
            num_intents=len(self.label_encoders["intent_encoder"]),
            num_categories=len(self.label_encoders["category_encoder"]),
            num_ner_labels=len(self.label_encoders["ner_label_encoder"]),
            dropout_rate=self.hyperparameters["dropout_rate"],
            fusion_type=self.fusion_type
        )

        if self.gpt2_tokenizer.pad_token_id is not None:
            self.classification_model.gpt2.resize_token_embeddings(len(self.gpt2_tokenizer))
        self.classification_model.load_state_dict(torch.load(model_path, map_location=self.device), strict=False)
        self.classification_model.to(self.device)
        self.classification_model.eval()

        try:
            self.generation_model = GPT2LMHeadModel.from_pretrained(self.generation_model_path).to(self.device)
            self.generation_tokenizer = GPT2Tokenizer.from_pretrained(self.generation_tokenizer_path)
            self.generation_tokenizer.pad_token = self.generation_tokenizer.eos_token
            self.generation_tokenizer.add_special_tokens({'additional_special_tokens': ['[INST]', '[RESP]', '[EOS]']})
            self.generation_model.resize_token_embeddings(len(self.generation_tokenizer))
        except Exception as e:
            print(f"Error loading generation model: {e}")
            print("Falling back to default GPT2...")
            self.generation_model = GPT2LMHeadModel.from_pretrained('gpt2').to(self.device)
            self.generation_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
            self.generation_tokenizer.pad_token = self.generation_tokenizer.eos_token
            self.generation_tokenizer.add_special_tokens({'additional_special_tokens': ['[INST]', '[RESP]', '[EOS]']})
            self.generation_model.resize_token_embeddings(len(self.generation_tokenizer))
        self.generation_model.eval()

    def process_input(self, instruction, session_id="default"):
        print(f"\nProcessing input: {instruction} for session {session_id} with {self.fusion_type} model")

        with self.lock:
            if session_id not in self.conversation_history:
                self.conversation_history[session_id] = []

            def run_classification():
                return inference_hybrid(
                    self.classification_model,
                    instruction,
                    self.gpt2_tokenizer,
                    self.distilbert_tokenizer,
                    self.label_encoders,
                    self.hyperparameters["max_length"],
                    self.device
                )

            classification_start = time.time()
            classification_result, classification_memory = get_peak_memory_usage(run_classification, device=self.device)
            classification_time = time.time() - classification_start

            intent = classification_result["intent"]["label"]
            intent_confidence = classification_result["intent"]["confidence"]
            category = classification_result["category"]["label"]
            category_confidence = classification_result["category"]["confidence"]
            entities = classification_result["ner"]
            entities_text = ", ".join([f"{entity['entity']} ({entity['label']})" for entity in entities]) if entities else "none"
            new_input = f"{instruction} [Classified: Intent is '{intent}' ({intent_confidence:.2f}), Category is '{category}' ({category_confidence:.2f}), Entities are {entities_text}]"

            def run_generation():
                return generate_response(
                    self.generation_model,
                    self.generation_tokenizer,
                    instruction,
                    classification_result,
                    self.conversation_history[session_id][-self.max_history:],
                    device=self.device
                )

            generation_start = time.time()
            generated_response, generation_memory = get_peak_memory_usage(run_generation, device=self.device)
            generation_time = time.time() - generation_start

            overall_time = classification_time + generation_time
            overall_memory = classification_memory + generation_memory

            self.conversation_history[session_id].append({
                "instruction": instruction,
                "response": generated_response
            })

        return {
            "instruction": instruction,
            "classified_input": new_input,
            "response": generated_response,
            "classification": {
                "intent": {"label": intent, "confidence": intent_confidence},
                "category": {"label": category, "confidence": category_confidence},
                "ner": entities
            },
            "classification_time": classification_time,
            "generation_time": generation_time,
            "overall_time": overall_time,
            "memory_usage": overall_memory
        }

    def clear_history(self, session_id="default"):
        with self.lock:
            if session_id in self.conversation_history:
                self.conversation_history[session_id] = []
                print(f"Conversation history cleared for session {session_id}")
            else:
                print(f"No history found for session {session_id}")

# Model paths for each architecture
model_paths = {
    "concat_model_dir": "../Hybrid_Concat_Freeze",
    "crossattention_model_dir": "../Hybrid_Cross_Attention_Freeze",
    "dense_model_dir": "../Hybrid_Dense_Layer_Freeze",
    "generation_model_path": "../text_generation_results_03-09-25/model",
    "generation_tokenizer_path": "../text_generation_results_03-09-25/tokenizer"
}

# Global service instances for each architecture
concat_service = ConversationalInferenceService(model_paths, fusion_type="concat")
crossattention_service = ConversationalInferenceService(model_paths, fusion_type="crossattention")
dense_service = ConversationalInferenceService(model_paths, fusion_type="dense")

# Helper function to process request
def process_request(service, data):
    if not data or 'message' not in data:
        return jsonify({"error": "No message provided"}), 400

    instruction = data['message']
    session_id = data.get('session_id', 'default')

    if instruction.lower() == "clear":
        service.clear_history(session_id)
        return jsonify({"response": "Conversation history cleared", "session_id": session_id})

    result = service.process_input(instruction, session_id)
    return jsonify({
        "response": result["response"],
        "classified_input": result["classified_input"],
        "session_id": session_id,
        "classification": {
            "intent": result["classification"]["intent"],
            "category": result["classification"]["category"],
            "ner": result["classification"]["ner"]
        },
        "metrics": {
            "classification_time": result["classification_time"],
            "generation_time": result["generation_time"],
            "overall_time": result["overall_time"],
            "memory_usage": result["memory_usage"]
        }
    })

# API Endpoints
@app.route('/concat', methods=['POST'])
def concat_chat():
    return process_request(concat_service, request.get_json())

@app.route('/crossattention', methods=['POST'])
def crossattention_chat():
    return process_request(crossattention_service, request.get_json())

@app.route('/dense', methods=['POST'])
def dense_chat():
    return process_request(dense_service, request.get_json())

if __name__ == "__main__":
    device_info = "CUDA" if torch.cuda.is_available() else "CPU"
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    print(f"Running on: {device_info}")
    app.run(host='0.0.0.0', port=5000, debug=True)