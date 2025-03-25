import json
import torch
import torch.nn as nn
from transformers import GPT2Config, GPT2Model, GPT2TokenizerFast, GPT2LMHeadModel, GPT2Tokenizer
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
    """Helper function for logging"""
    print(message)

# Baseline GPT-2 Multi-Task Model
class BaselineGPT2MultiTask(nn.Module):
    def __init__(self, num_intents: int, num_categories: int, num_ner_labels: int,
                 dropout_rate: float, loss_weights: Optional[Dict[str, float]] = None,
                 ner_class_weights: Optional[torch.Tensor] = None,
                 category_class_weights: Optional[torch.Tensor] = None,
                 intent_class_weights: Optional[torch.Tensor] = None):
        super().__init__()
        log_to_file("Initializing classification model...")
        self.config = GPT2Config.from_pretrained('gpt2')
        self.gpt2 = GPT2Model.from_pretrained('gpt2')
        hidden_size = self.config.n_embd

        for param in self.gpt2.parameters():
            param.requires_grad = False
        log_to_file("All GPT-2 layers remain frozen")

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

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                intent_labels: Optional[torch.Tensor] = None,
                category_labels: Optional[torch.Tensor] = None,
                ner_labels: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        outputs = self.gpt2(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        masked_features = sequence_output * attention_mask.unsqueeze(-1)
        sequence_repr = masked_features.sum(dim=1) / attention_mask.sum(dim=1, keepdim=True)

        intent_logits = self.intent_head(sequence_repr)
        category_logits = self.category_head(sequence_repr)
        ner_logits = self.ner_head(sequence_output)

        output_dict = {
            'intent_logits': intent_logits,
            'category_logits': category_logits,
            'ner_logits': ner_logits
        }

        if all(label is not None for label in [intent_labels, category_labels, ner_labels]):
            intent_loss = self.intent_loss_fn(intent_logits, intent_labels)
            category_loss = self.category_loss_fn(category_logits, category_labels)
            active_loss = attention_mask.view(-1) == 1
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

def inference_baseline(model, text, tokenizer, label_encoders, max_length, device):
    """Run inference with the BaselineGPT2MultiTask model"""
    model.eval()
    encoding = tokenizer(text, max_length=max_length, padding="max_length", truncation=True, return_tensors="pt")

    inputs = {
        "input_ids": encoding["input_ids"].to(device),
        "attention_mask": encoding["attention_mask"].to(device)
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
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0].tolist())
    seq_len = int(inputs["attention_mask"][0].sum().item())
    ner_labels = [ner_decoder[pred] for pred in ner_preds[:seq_len]]

    entities = []
    current_entity = None
    entity_tokens = []
    entity_confidences = []
    entity_type = None

    for i, (token, label, confidence) in enumerate(zip(tokens[:seq_len], ner_labels, ner_confidences[:seq_len])):
        if label.startswith("B-"):
            if current_entity is not None:
                entity_text = tokenizer.convert_tokens_to_string(entity_tokens).strip()
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
            entity_text = tokenizer.convert_tokens_to_string(entity_tokens).strip()
            if entity_text:
                avg_confidence = sum(entity_confidences) / len(entity_confidences)
                entities.append({"entity": entity_text, "label": entity_type, "confidence": avg_confidence})
            current_entity = None
            entity_tokens = []
            entity_confidences = []
            entity_type = None

    if current_entity is not None:
        entity_text = tokenizer.convert_tokens_to_string(entity_tokens).strip()
        if entity_text:
            avg_confidence = sum(entity_confidences) / len(entity_confidences)
            entities.append({"entity": entity_text, "label": entity_type, "confidence": avg_confidence})

    return {
        "intent": {"label": intent_label, "confidence": intent_confidence},
        "category": {"label": category_label, "confidence": category_confidence},
        "ner": entities
    }

def generate_response(model, tokenizer, instruction, classification, history, max_length=512, device="cuda"):
    """Generate a response using GPT-2"""
    model.eval()
    intent = classification["intent"]["label"]
    category = classification["category"]["label"]
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
    return mem_info.rss / 1024 / 1024  # Convert bytes to MB

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

class BaselineInferenceService:
    def __init__(self, model_paths):
        self.output_dir = model_paths["baseline_model_dir"]
        self.generation_model_path = model_paths["generation_model_path"]
        self.generation_tokenizer_path = model_paths["generation_tokenizer_path"]
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.conversation_history = {}
        self.max_history = 5
        self.lock = Lock()
        self.load_models()

    def load_models(self):
        print(f"\nLoading baseline model on {self.device.upper()}...")
        encoders_path = os.path.join(self.output_dir, "label_encoders.json")
        hyperparams_path = os.path.join(self.output_dir, "hyperparameters.json")
        model_path = os.path.join(self.output_dir, "baseline_model.pth")

        with open(encoders_path, 'r', encoding='utf-8') as f:
            self.label_encoders = json.load(f)
        with open(hyperparams_path, 'r', encoding='utf-8') as f:
            self.hyperparameters = json.load(f)

        self.gpt2_tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
        if self.gpt2_tokenizer.pad_token is None:
            self.gpt2_tokenizer.add_special_tokens({'pad_token': '[PAD]'})

        self.classification_model = BaselineGPT2MultiTask(
            num_intents=len(self.label_encoders["intent_encoder"]),
            num_categories=len(self.label_encoders["category_encoder"]),
            num_ner_labels=len(self.label_encoders["ner_label_encoder"]),
            dropout_rate=self.hyperparameters["dropout_rate"]
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
        print(f"\nProcessing input: {instruction} for session {session_id} with baseline model")

        with self.lock:
            if session_id not in self.conversation_history:
                self.conversation_history[session_id] = []

            def run_classification():
                return inference_baseline(
                    self.classification_model,
                    instruction,
                    self.gpt2_tokenizer,
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

# Model paths
model_paths = {
    "baseline_model_dir": "../Baseline_freeze_v1",
    "generation_model_path": "../text_generation_results_03-09-25/model",
    "generation_tokenizer_path": "../text_generation_results_03-09-25/tokenizer"
}

# Global service instance
baseline_service = BaselineInferenceService(model_paths)

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

# API Endpoint
@app.route('/baseline', methods=['POST'])
def baseline_chat():
    return process_request(baseline_service, request.get_json())

if __name__ == "__main__":
    device_info = "CUDA" if torch.cuda.is_available() else "CPU"
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    print(f"Running on: {device_info}")
    app.run(host='0.0.0.0', port=5001, debug=True)  # Changed port to 5001 to avoid conflict with hybrid