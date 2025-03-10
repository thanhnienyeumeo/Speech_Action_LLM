import torch
import torch.nn as nn
from transformers import AutoModel, AutoModelForCausalLM, WhisperModel, Wav2Vec2Model
from transformers import AutoProcessor, AutoTokenizer
from transformers import Trainer, TrainingArguments
from dataclasses import dataclass
from typing import Dict, List, Optional, Union, Tuple
import torch.nn.functional as F


class MultiModalProjector(nn.Module):
    """Projects audio embeddings to text embedding space"""
    
    def __init__(self, in_dim, out_dim, hidden_dim=None):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = out_dim
            
        self.linear1 = nn.Linear(in_dim, hidden_dim, bias=True)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_dim, out_dim, bias=True)
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using Kaiming initialization for ReLU activation"""
        # Initialize first layer
        nn.init.kaiming_normal_(self.linear1.weight, mode='fan_out', nonlinearity='relu')
        nn.init.zeros_(self.linear1.bias)
        
        # Initialize second layer - use smaller values for the output projection
        nn.init.kaiming_normal_(self.linear2.weight, mode='fan_out', nonlinearity='relu')
        nn.init.zeros_(self.linear2.bias)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x


class SpeechToTextModel(nn.Module):
    def __init__(
        self,
        audio_encoder_id="openai/whisper-small",
        lm_id="gpt2",
        use_pooler=True,
        projection_hidden_size=None,
        freeze_encoder=False,
        freeze_lm=False,
    ):
        super().__init__()
        
        # Initialize audio encoder
        # print(f"Loading audio encoder: {audio_encoder_id}")
        # if "whisper" in audio_encoder_id:
        if "whisper" in audio_encoder_id.lower():
            # print('whisper load')
            self.audio_encoder = WhisperModel.from_pretrained(audio_encoder_id, torch_dtype = torch.bfloat16)
            self.audio_dim = self.audio_encoder.config.d_model  # Usually 512 for small
        else:
            self.audio_encoder = Wav2Vec2Model.from_pretrained(audio_encoder_id, torch_dtype = torch.bfloat16)
            self.audio_dim = self.audio_encoder.config.hidden_size
            
        # Initialize LLM
        print(f"Loading language model: {lm_id}")
        self.lm = AutoModelForCausalLM.from_pretrained(lm_id,
                                                       torch_dtype = torch.bfloat16)
        self.lm_dim = self.lm.config.hidden_size
        
        # Initialize audio processor and LM tokenizer
        self.processor = AutoProcessor.from_pretrained(audio_encoder_id)
        self.tokenizer = AutoTokenizer.from_pretrained(lm_id)
        
        # Add special tokens if needed
        special_tokens = ["<|audio|>", "<|endofaudio|>"]
        for id in range(1,27):
            special_tokens.append(f"<|{id}|>")
        num_added = self.tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
        if num_added > 0:
            self.lm.resize_token_embeddings(len(self.tokenizer))
        
        self.audio_token_id = self.tokenizer.convert_tokens_to_ids("<|audio|>")
        self.audio_end_token_id = self.tokenizer.convert_tokens_to_ids("<|endofaudio|>")
        
        # Average pooling for audio features
        self.use_pooler = use_pooler
        self.audio_avg_pooler = nn.AvgPool1d(kernel_size=2, stride=2, padding=0)
        
        # Projection layer from audio space to LM space
        self.audio_projection_layer = MultiModalProjector(
            in_dim=self.audio_dim,
            out_dim=self.lm_dim,
            hidden_dim=projection_hidden_size
        )
        #4628159*9427831
        # Freeze components if specified
        if freeze_encoder:
            for param in self.audio_encoder.parameters():
                param.requires_grad = False
        #921480(197+15)
        if freeze_lm:
            for param in self.lm.parameters():
                param.requires_grad = False
    
    # def _prepare_audio_features(self, audio_features):
    #     """Process audio features for LM consumption"""
        
    #     # Apply average pooling to reduce sequence length if needed
    #     if self.use_pooler:
    #         # Transpose to (batch, channels, time)
    #         audio_features = audio_features.transpose(1, 2)
    #         audio_features = self.audio_avg_pooler(audio_features)
    #         # Transpose back to (batch, time, channels)
    #         audio_features = audio_features.transpose(1, 2)
            
    #     # Project to LM dimension
    #     audio_embeddings = self.audio_projection_layer(audio_features)
        
    #     return audio_embeddings

    # def _prepare_audio_features(self, audio_features):
    #     """Process audio features for LM consumption"""
        
    #     # Handle 4D input (batch, channels, freq, time) from audio models
    #     if len(audio_features.shape) == 4:
    #         # Reshape to (batch, time, channels*freq)
    #         batch, channels, freq, time = audio_features.shape
    #         audio_features = audio_features.permute(0, 3, 1, 2).reshape(batch, time, channels*freq)
        
    #     # Apply average pooling to reduce sequence length if needed
    #     if self.use_pooler:
    #         # Transpose to (batch, channels, time)
    #         audio_features = audio_features.transpose(1, 2)
    #         audio_features = self.audio_avg_pooler(audio_features)
    #         # Transpose back to (batch, time, channels)
    #         audio_features = audio_features.transpose(1, 2)
            
    #     # Project to LM dimension
    #     audio_embeddings = self.audio_projection_layer(audio_features)
        
    #     return audio_embeddings

    # def _prepare_audio_features(self, audio_features):
    #     """Process audio features for LM consumption"""
        
    #     # Handle 4D input (batch, channels, freq, time) from Whisper models
    #     if len(audio_features.shape) == 4:
    #         # Reshape to (batch, time, channels*freq)
    #         batch, channels, freq, time = audio_features.shape
    #         audio_features = audio_features.permute(0, 3, 1, 2).reshape(batch, time, channels*freq)
        
    #     # Apply average pooling to reduce sequence length if needed
    #     if self.use_pooler:
    #         # Transpose to (batch, channels, time)
    #         audio_features = audio_features.transpose(1, 2)
    #         audio_features = self.audio_avg_pooler(audio_features)
    #         # Transpose back to (batch, time, channels)
    #         audio_features = audio_features.transpose(1, 2)
            
    #     # Project to LM dimension
    #     audio_embeddings = self.audio_projection_layer(audio_features)
        
    #     return audio_embeddings

    def _prepare_audio_features(self, audio_features):
        """Process audio features for LM consumption"""
        
        # Print shape for debugging
        # print(f"Audio features shape: {audio_features.shape}")
        
        # For Whisper models, skip the pooling and reshaping if we already have 3D
        if len(audio_features.shape) == 3:
            # Apply pooling if needed
            if self.use_pooler:
                # Transpose to (batch, channels, time)
                audio_features = audio_features.transpose(1, 2)
                audio_features = self.audio_avg_pooler(audio_features)
                # Transpose back to (batch, time, channels)
                audio_features = audio_features.transpose(1, 2)
        
        # Handle 4D input (should be rare with the encode_audio fix)
        elif len(audio_features.shape) == 4:
            # Print warning since we should not get here with the fixed encoder
            print("Warning: Got 4D tensor in _prepare_audio_features")
            # Reshape to (batch, time, channels*freq)
            batch, channels, freq, time = audio_features.shape
            audio_features = audio_features.permute(0, 3, 1, 2).reshape(batch, time, channels*freq)
            
            # Apply pooling if needed
            if self.use_pooler:
                # Transpose to (batch, channels, time)
                audio_features = audio_features.transpose(1, 2)
                audio_features = self.audio_avg_pooler(audio_features)
                # Transpose back to (batch, time, channels)
                audio_features = audio_features.transpose(1, 2)
        
        # Project to LM dimension
        audio_embeddings = self.audio_projection_layer(audio_features)
        
        return audio_embeddings
    
    # def encode_audio(self, audio_input, attention_mask=None):
    #     """Encode audio using the audio encoder"""
        
    #     # Process with audio encoder
    #     with torch.no_grad() if self.audio_encoder.training == False else torch.enable_grad():
    #         if "whisper" in self.audio_encoder.config._name_or_path:
    #             encoder_outputs = self.audio_encoder.encoder(
    #                 audio_input,
    #                 attention_mask=attention_mask,
    #                 return_dict=True
    #             )
    #             audio_features = encoder_outputs.last_hidden_state
    #         else:
    #             encoder_outputs = self.audio_encoder(
    #                 audio_input,
    #                 attention_mask=attention_mask,
    #                 return_dict=True
    #             )
    #             audio_features = encoder_outputs.last_hidden_state
                
    #     return audio_features
    # 
    
    # def encode_audio(self, audio_input, attention_mask=None):
    #     """Encode audio using the audio encoder"""
        
    #     # Process with audio encoder
    #     with torch.no_grad() if not self.audio_encoder.training else torch.enable_grad():
    #         if "whisper" in self.audio_encoder.config._name_or_path:
    #             # For Whisper, we need to use the full model properly
    #             # This will handle the 4D input correctly
    #             encoder_outputs = self.audio_encoder(
    #                 input_features=audio_input,
    #                 attention_mask=attention_mask,
    #                 return_dict=True
    #             ).encoder_last_hidden_state
                
    #             # The encoder output is now 3D: [batch_size, sequence_length, hidden_size]
    #             audio_features = encoder_outputs
    #         else:
    #             # For other models like Wav2Vec2
    #             encoder_outputs = self.audio_encoder(
    #                 audio_input,
    #                 attention_mask=attention_mask,
    #                 return_dict=True
    #             )
    #             audio_features = encoder_outputs.last_hidden_state
                    
    #     return audio_features

    # def encode_audio(self, audio_input, attention_mask=None):
    #     """Encode audio using the audio encoder"""
        
    #     # Process with audio encoder
    #     with torch.no_grad() if not self.audio_encoder.training else torch.enable_grad():
    #         if "whisper" in self.audio_encoder.config._name_or_path:
    #             # For Whisper, we need to use the model differently 
    #             # Whisper expects input shape of [batch, 1, 80, 3000]
                
    #             # Access the encoder part of the WhisperModel
    #             encoder_outputs = self.audio_encoder.get_encoder()(
    #                 input_features=audio_input,
    #                 attention_mask=attention_mask,
    #             )
                
    #             # The encoder output should now be 3D: [batch_size, sequence_length, hidden_size]
    #             audio_features = encoder_outputs.last_hidden_state
                
    #             # Verify the shape is 3D and reshape if needed
    #             if len(audio_features.shape) == 4:
    #                 # Reshape 4D output to 3D
    #                 batch, channels, freq, time = audio_features.shape
    #                 audio_features = audio_features.permute(0, 3, 1, 2).reshape(batch, time, channels*freq)
    #         else:
    #             # For other models like Wav2Vec2
    #             encoder_outputs = self.audio_encoder(
    #                 audio_input,
    #                 attention_mask=attention_mask,
    #                 return_dict=True
    #             )
    #             audio_features = encoder_outputs.last_hidden_state
                    
    #     return audio_features

    # def encode_audio(self, audio_input, attention_mask=None):
    #     """Encode audio using the audio encoder"""
        
    #     # Process with audio encoder
    #     with torch.no_grad() if not self.audio_encoder.training else torch.enable_grad():
    #         if "whisper" in self.audio_encoder.config._name_or_path or "Whisper" in self.audio_encoder.config._name_or_path:
    #             # Print shape for debugging
    #             print(f"Audio input shape: {audio_input.shape}")
                
    #             # For Whisper, bypass the encoder and directly use the encoder_hidden_states
    #             # This avoids the 4D tensor issue
    #             if len(audio_input.shape) == 4:  # [batch, channels, mel_bins, time]
    #             # Whisper expects this shape, so we use the encoder directly
    #                 encoder_outputs = self.audio_encoder.encoder(
    #                     input_features=audio_input,
    #                     attention_mask=attention_mask,
    #                     return_dict=True
    #                 )
    #                 audio_features = encoder_outputs.last_hidden_state
    #                 print(f"Encoder output shape: {audio_features.shape}")
    #             else:
    #                 raise ValueError(f"Unexpected input shape for Whisper: {audio_input.shape}, expected 4D tensor")
    #             # encoder_outputs = self.audio_encoder.get_encoder()(
    #             #     input_features=audio_input,
    #             #     attention_mask=attention_mask,
    #             #     return_dict=True
    #             # )
                
    #             # # Get the last hidden state, which is already in 3D format
    #             # audio_features = encoder_outputs.last_hidden_state
    #             # print(f"Encoder output shape: {audio_features.shape}")
                
    #         else:
    #             # For other models like Wav2Vec2
    #             encoder_outputs = self.audio_encoder(
    #                 audio_input,
    #                 attention_mask=attention_mask,
    #                 return_dict=True
    #             )
    #             audio_features = encoder_outputs.last_hidden_state
                    
    #     return audio_features

    def encode_audio(self, audio_input, attention_mask=None):
        """Encode audio using the audio encoder"""
        
        # Process with audio encoder
        with torch.no_grad() if not self.audio_encoder.training else torch.enable_grad():
            if "whisper" in self.audio_encoder.config._name_or_path.lower():
                # Print shape for debugging
                # print(f"Audio input shape: {audio_input.shape}")
                
                # Reshape input for Whisper encoder - it expects 3D input not 4D
                if len(audio_input.shape) == 4:
                    # Convert from [batch, 1, mel_bins, frames] to [batch, mel_bins, frames]
                    # Whisper encoder expects [batch, feature_size, sequence_length]
                    audio_input = audio_input.squeeze(1)  # Remove channel dimension
                    # print(f"Squeezed input shape: {audio_input.shape}")
                
                # Now process with the encoder
                encoder_outputs = self.audio_encoder.encoder(
                    input_features=audio_input,
                    attention_mask=attention_mask,
                    return_dict=True
                )
                audio_features = encoder_outputs.last_hidden_state
                # print(f"Encoder output shape: {audio_features.shape}")
                
            else:
                # For other models like Wav2Vec2
                encoder_outputs = self.audio_encoder(
                    audio_input,
                    attention_mask=attention_mask,
                    return_dict=True
                )
                audio_features = encoder_outputs.last_hidden_state
                    
        return audio_features
    
    # def forward(
    #     self,
    #     audio_input=None,
    #     audio_attention_mask=None,
    #     input_ids=None,
    #     attention_mask=None,
    #     labels=None,
    #     replace_audio_token=True,
    # ):
    #     batch_size = input_ids.shape[0] if input_ids is not None else audio_input.shape[0]
    #     device = input_ids.device if input_ids is not None else audio_input.device
        
    #     # Process audio if provided
    #     audio_embeddings = None
    #     if audio_input is not None:
    #         # Encode audio
    #         audio_features = self.encode_audio(audio_input, audio_attention_mask)
            
    #         # Project to LM space
    #         audio_embeddings = self._prepare_audio_features(audio_features)
        
    #     # If replacing audio token with actual audio embeddings
    #     if replace_audio_token and input_ids is not None and audio_embeddings is not None:
    #         # Find all audio tokens and end tokens
    #         audio_token_indices = torch.where(input_ids == self.audio_token_id)
    #         audio_end_indices = torch.where(input_ids == self.audio_end_token_id)
            
    #         # Create a new input embeddings tensor
    #         input_shape = input_ids.shape
    #         embedded_input = self.lm.transformer.wte(input_ids)
            
    #         # Replace audio tokens with audio embeddings
    #         for batch_idx in range(batch_size):
    #             batch_audio_positions = torch.where(audio_token_indices[0] == batch_idx)[0]
    #             batch_end_positions = torch.where(audio_end_indices[0] == batch_idx)[0]
                
    #             if len(batch_audio_positions) > 0 and len(batch_end_positions) > 0:
    #                 start_pos = audio_token_indices[1][batch_audio_positions[0]]
    #                 end_pos = audio_end_indices[1][batch_end_positions[0]]
                    
    #                 # Get audio embeddings for this example
    #                 audio_emb = audio_embeddings[batch_idx]
                    
    #                 # Make sure we don't overflow
    #                 audio_length = min(end_pos - start_pos - 1, audio_emb.shape[0])
                    
    #                 # Replace the text embeddings with audio embeddings
    #                 if audio_length > 0:
    #                     embedded_input[batch_idx, start_pos+1:start_pos+1+audio_length] = audio_emb[:audio_length]
            
    #         # Pass the embedded input to the LM
    #         outputs = self.lm(
    #             inputs_embeds=embedded_input,
    #             attention_mask=attention_mask,
    #             labels=labels,
    #             return_dict=True
    #         )
    #     else:
    #         # Standard LM forward pass
    #         outputs = self.lm(
    #             input_ids=input_ids,
    #             attention_mask=attention_mask,
    #             labels=labels,
    #             return_dict=True
    #         )
        
    #     return outputs
    def forward(
    self,
    audio_input=None,
    audio_attention_mask=None,
    input_ids=None,
    attention_mask=None,
    labels=None,
    replace_audio_token=True,
):
        batch_size = input_ids.shape[0] if input_ids is not None else audio_input.shape[0]
        device = input_ids.device if input_ids is not None else audio_input.device
        
        # Process audio if provided
        audio_embeddings = None
        if audio_input is not None:
            # Encode audio
            audio_features = self.encode_audio(audio_input, audio_attention_mask)
            
            # Project to LM space
            audio_embeddings = self._prepare_audio_features(audio_features)
        
        # If replacing audio token with actual audio embeddings
        if replace_audio_token and input_ids is not None and audio_embeddings is not None:
            # Find all audio tokens and end tokens
            audio_token_indices = torch.where(input_ids == self.audio_token_id)
            audio_end_indices = torch.where(input_ids == self.audio_end_token_id)
            
            # Create a new input embeddings tensor - handle different model architectures
            input_shape = input_ids.shape
            
            # Handle different model architectures for embedding lookup
            if hasattr(self.lm, 'transformer') and hasattr(self.lm.transformer, 'wte'):
                # GPT-2 style models
                embedded_input = self.lm.transformer.wte(input_ids)
            elif hasattr(self.lm, 'model') and hasattr(self.lm.model, 'embed_tokens'):
                # Newer models like Qwen2
                embedded_input = self.lm.model.embed_tokens(input_ids)
            else:
                # Fallback - try common paths
                if hasattr(self.lm, 'get_input_embeddings'):
                    embedded_input = self.lm.get_input_embeddings()(input_ids)
                else:
                    raise ValueError(f"Could not find embedding layer for model type: {type(self.lm)}")
            
            # Replace audio tokens with audio embeddings
            for batch_idx in range(batch_size):
                batch_audio_positions = torch.where(audio_token_indices[0] == batch_idx)[0]
                batch_end_positions = torch.where(audio_end_indices[0] == batch_idx)[0]
                
                if len(batch_audio_positions) > 0 and len(batch_end_positions) > 0:
                    start_pos = audio_token_indices[1][batch_audio_positions[0]]
                    end_pos = audio_end_indices[1][batch_end_positions[0]]
                    
                    # Get audio embeddings for this example
                    audio_emb = audio_embeddings[batch_idx]
                    
                    # Make sure we don't overflow
                    audio_length = min(end_pos - start_pos - 1, audio_emb.shape[0])
                    
                    # Replace the text embeddings with audio embeddings
                    if audio_length > 0:
                        embedded_input[batch_idx, start_pos+1:start_pos+1+audio_length] = audio_emb[:audio_length]
            
            # Pass the embedded input to the LM
            outputs = self.lm(
                inputs_embeds=embedded_input,
                attention_mask=attention_mask,
                labels=labels,
                return_dict=True
            )
        else:
            # Standard LM forward pass
            outputs = self.lm(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                return_dict=True
            )
        
        return outputs

    def generate_from_audio(
        self, 
        audio_input, 
        audio_attention_mask=None,
        prompt="Transcription: <|audio|><|endofaudio|>",
        max_length=100,
        **generate_kwargs
    ):
        """Generate text from audio input"""
        
        # Tokenize prompt
        prompt_tokens = self.tokenizer(prompt, return_tensors="pt").to(audio_input.device)
        
        # Encode audio
        audio_features = self.encode_audio(audio_input, audio_attention_mask)
        
        # Project to LM space
        audio_embeddings = self._prepare_audio_features(audio_features)
        
        # Find tokens to replace with audio embeddings
        input_ids = prompt_tokens.input_ids
        batch_size = audio_input.shape[0]
        
        # Create attention mask if not provided
        if "attention_mask" not in prompt_tokens:
            attention_mask = torch.ones_like(input_ids)
        else:
            attention_mask = prompt_tokens.attention_mask
        
        # Find all audio tokens and end tokens
        audio_token_indices = torch.where(input_ids == self.audio_token_id)
        audio_end_indices = torch.where(input_ids == self.audio_end_token_id)
        
        # Create input embeddings
        if hasattr(self.lm, 'transformer') and hasattr(self.lm.transformer, 'wte'):
        # GPT-2 style models
            embedded_input = self.lm.transformer.wte(input_ids)
        elif hasattr(self.lm, 'model') and hasattr(self.lm.model, 'embed_tokens'):
        # Newer models like Qwen2
            embedded_input = self.lm.model.embed_tokens(input_ids)
        else:
        # Fallback - try common paths
            if hasattr(self.lm, 'get_input_embeddings'):
                embedded_input = self.lm.get_input_embeddings()(input_ids)
            else:
                raise ValueError(f"Could not find embedding layer for model type: {type(self.lm)}")
        
        # Replace audio tokens with audio embeddings
        for batch_idx in range(batch_size):
            batch_audio_positions = torch.where(audio_token_indices[0] == batch_idx)[0]
            batch_end_positions = torch.where(audio_end_indices[0] == batch_idx)[0]
            
            if len(batch_audio_positions) > 0 and len(batch_end_positions) > 0:
                start_pos = audio_token_indices[1][batch_audio_positions[0]]
                end_pos = audio_end_indices[1][batch_end_positions[0]]
                
                # Get audio embeddings for this example
                audio_emb = audio_embeddings[batch_idx]
                
                # Make sure we don't overflow
                audio_length = min(end_pos - start_pos - 1, audio_emb.shape[0])
                
                if audio_length > 0:
                    embedded_input[batch_idx, start_pos+1:start_pos+1+audio_length] = audio_emb[:audio_length]
        
        # Generate text
        outputs = self.lm.generate(
            inputs_embeds=embedded_input,
            attention_mask=attention_mask,
            max_length=max_length,
            **generate_kwargs
        )
        
        # Decode outputs
        decoded_outputs = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        
        return decoded_outputs
    @classmethod
    def from_pretrained(cls, model_dir):
        """Load model from directory"""
        import os
        import json
        
        # Load config
        with open(os.path.join(model_dir, "model_config.json"), "r") as f:
            config = json.load(f)
        
        # Initialize model with saved components
        model = cls(
            audio_encoder_id=os.path.join(model_dir, "audio_encoder"),
            lm_id=os.path.join(model_dir, "lm"),
            use_pooler=config["use_pooler"]
        )
        
        # Load projection layer weights
        projection_path = os.path.join(model_dir, "projection_layer.pt")
        model.audio_projection_layer.load_state_dict(torch.load(projection_path))
        
        return model

def prepare_model_for_lora(model, lora_config):
    """Prepares the model for LoRA fine-tuning by:
    1. Adding LoRA weights to specified layers
    2. Freezing non-LoRA weights
    """
    # Apply LoRA to LLM component
    model.lm = get_peft_model(model.lm, lora_config)
    
    # Freeze all parameters except LoRA parameters
    for param in model.parameters():
        param.requires_grad = False
        
    # Unfreeze projection layer if needed for cross-modal adaptation
    for param in model.audio_projection_layer.parameters():
        param.requires_grad = True
    
    # Unfreeze LoRA parameters
    for name, param in model.lm.named_parameters():
        if 'lora' in name:
            param.requires_grad = True
            
    # Count trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable_params:,} ({100 * trainable_params / total_params:.2f}% of {total_params:,})")
    
    return model

# Create data processing class for training
# @dataclass
# class SpeechToTextDataCollator:
#     """Data collator for speech-to-text models"""
    
#     processor: AutoProcessor
#     tokenizer: AutoTokenizer
#     max_length: int = 512
    
#     def __call__(self, features):
#         # Process audio
#         audio_inputs = [feature["audio"] for feature in features]
#         audio_processed = self.processor(
#             audio=audio_inputs,
#             padding="longest",
#             return_tensors="pt",
#             sampling_rate=16000
#         )
        
#         # Process text
#         text_inputs = [f"Transcription: <|audio|><|endofaudio|> {feature['text']}" for feature in features]
#         text_processed = self.tokenizer(
#             text_inputs,
#             padding="max_length",
#             truncation=True,
#             max_length=self.max_length,
#             return_tensors="pt"
#         )
        
#         # Create labels (shifted input_ids)
#         labels = text_processed.input_ids.clone()
#         labels[labels == self.tokenizer.pad_token_id] = -100  # Ignore padding in loss
        
#         return {
#             "audio_input": audio_processed.input_features,
#             "audio_attention_mask": audio_processed.attention_mask,
#             "input_ids": text_processed.input_ids,
#             "attention_mask": text_processed.attention_mask,
#             "labels": labels
#         }


# @dataclass
# class SpeechToTextDataCollator:
#     """Data collator for speech-to-text models"""
    
#     processor: AutoProcessor
#     tokenizer: AutoTokenizer
#     max_length: int = 512
    
#     def __call__(self, features):
#         # Process audio
#         audio_inputs = [feature["audio"] for feature in features]
        
#         # Process with Whisper's expected parameters
#         audio_processed = self.processor(
#             audio=audio_inputs,
#             padding="max_length",  # Change to max_length to ensure we reach 3000 frames
#             max_length=3000,       # Whisper expects 3000 frames
#             return_tensors="pt",
#             sampling_rate=16000,
#             return_attention_mask=True  # Explicitly request attention mask
#         )
        
#         # Handle case where attention_mask is not returned
#         if hasattr(audio_processed, 'attention_mask'):
#             audio_attention_mask = audio_processed.attention_mask
#         else:
#             print('No attention mask returned, creating default mask')
#             # Create default attention mask (all 1s for real content, 0s for padding)
#             audio_attention_mask = torch.ones(audio_processed.input_features.shape[0], 
#                                               audio_processed.input_features.shape[1],
#                                               dtype=torch.long)
        
#         # Process text
#         text_inputs = [f"Transcription: <|audio|><|endofaudio|> {feature['text']}" for feature in features]
#         text_processed = self.tokenizer(
#             text_inputs,
#             padding="max_length",
#             truncation=True,
#             max_length=self.max_length,
#             return_tensors="pt"
#         )
        
#         # Create labels (shifted input_ids)
#         labels = text_processed.input_ids.clone()
#         labels[labels == self.tokenizer.pad_token_id] = -100  # Ignore padding in loss
        
#         return {
#             "audio_input": audio_processed.input_features,
#             "audio_attention_mask": audio_attention_mask,
#             "input_ids": text_processed.input_ids,
#             "attention_mask": text_processed.attention_mask,
#             "labels": labels
#         }

# @dataclass
# class SpeechToTextDataCollator:
#     """Data collator for speech-to-text models"""
    
#     processor: AutoProcessor
#     tokenizer: AutoTokenizer
#     max_length: int = 512
    
#     def __call__(self, features):
#         # Process audio
#         audio_inputs = [feature["audio"] for feature in features]
        
#         # For Whisper models, we need to process in two steps:
#         # 1. First convert audio to log-mel spectrograms
#         # 2. Then ensure they're padded to exactly 3000 frames
        
#         # Step 1: Convert to log-mel spectrograms
#         inputs = self.processor.feature_extractor(audio_inputs, sampling_rate=16000, return_tensors="pt")
#         input_features = inputs.input_features
        
#         # Step 2: Pad or truncate to exactly 3000 frames
#         batch_size, num_channels, num_frames, num_mels = input_features.shape
#         target_length = 3000
        
#         if num_frames < target_length:
#             # Pad if shorter
#             padding = torch.zeros(batch_size, num_channels, target_length - num_frames, num_mels)
#             input_features = torch.cat([input_features, padding], dim=2)
#         elif num_frames > target_length:
#             # Truncate if longer
#             input_features = input_features[:, :, :target_length, :]
        
#         # Create appropriate attention mask
#         audio_attention_mask = torch.ones(batch_size, target_length, dtype=torch.long)
#         for i, audio in enumerate(audio_inputs):
#             # Set mask to 0 for padded regions
#             actual_length = min(num_frames, target_length)
#             audio_attention_mask[i, actual_length:] = 0
        
#         # Process text
#         text_inputs = [f"Transcription: <|audio|><|endofaudio|> {feature['text']}" for feature in features]
#         text_processed = self.tokenizer(
#             text_inputs,
#             padding="max_length",
#             truncation=True,
#             max_length=self.max_length,
#             return_tensors="pt"
#         )
        
#         # Create labels (shifted input_ids)
#         labels = text_processed.input_ids.clone()
#         labels[labels == self.tokenizer.pad_token_id] = -100  # Ignore padding in loss
        
#         return {
#             "audio_input": input_features,
#             "audio_attention_mask": audio_attention_mask,
#             "input_ids": text_processed.input_ids,
#             "attention_mask": text_processed.attention_mask,
#             "labels": labels
#         }

@dataclass
class SpeechToTextDataCollator:
    """Data collator for speech-to-text models"""
    
    processor: AutoProcessor
    tokenizer: AutoTokenizer
    max_length: int = 512
    
    def __call__(self, features):
        # Process audio
        audio_inputs = [feature["audio"] for feature in features]
        
        # Step 1: Convert to features using the processor's feature extractor
        inputs = self.processor.feature_extractor(audio_inputs, sampling_rate=16000, return_tensors="pt")
        input_features = inputs.input_features.to(dtype=torch.bfloat16)
        
        # Step 2: Check shape and pad appropriately for Whisper
        # Whisper expects shape (batch_size, 1, 80, 3000) - (batch, channels, mel_bins, frames)
        shape = input_features.shape
        batch_size = shape[0]
        target_length = 3000
        
        # Handle different possible shapes from the feature extractor
        if len(shape) == 4:  # (batch, channels, mel_bins, frames)
            num_channels, num_mels, num_frames = shape[1], shape[2], shape[3]
            print('needed change')
            if num_frames < target_length:
                # Pad if shorter
                padding = torch.zeros(batch_size, num_channels, num_mels, target_length - num_frames, dtype = torch.bfloat16)
                input_features = torch.cat([input_features, padding], dim=3)
            elif num_frames > target_length:
                # Truncate if longer
                input_features = input_features[:, :, :, :target_length]
                
        elif len(shape) == 3:  # (batch, frames, features) or (batch, channels, frames)
            # Determine which dimension is the time dimension (usually the middle one for audio)
            # For Whisper, feature dimension is typically 80 (mel bins)
            if shape[1] == 80:  # If middle dim is 80, it's likely (batch, mel_bins, frames)
                num_frames = shape[2]
                if num_frames < target_length:
                    padding = torch.zeros(batch_size, shape[1], target_length - num_frames)
                    input_features = torch.cat([input_features, padding], dim=2)
                elif num_frames > target_length:
                    input_features = input_features[:, :, :target_length]
                    
                # Reshape to Whisper's expected 4D format
                input_features = input_features.unsqueeze(1)  # Add channel dimension
                
            else:  # Assume it's (batch, frames, features)
                num_frames = shape[1]
                if num_frames < target_length:
                    padding = torch.zeros(batch_size, target_length - num_frames, shape[2])
                    input_features = torch.cat([input_features, padding], dim=1)
                elif num_frames > target_length:
                    input_features = input_features[:, :target_length, :]
                
                # Reshape to Whisper's expected format (batch, 1, features, frames)
                input_features = input_features.transpose(1, 2).unsqueeze(1)
        
        # Create appropriate attention mask (based on the target shape now)
        audio_attention_mask = torch.ones(batch_size, target_length, dtype=torch.long)
        # Set the mask to 0 for padded regions
        for i, audio in enumerate(audio_inputs):
            # Estimate the actual length (in frames)
            actual_length = min(input_features.shape[-1] if len(shape) == 3 else shape[-1], target_length)
            audio_attention_mask[i, actual_length:] = 0
        
        # Process text
        text_inputs = [f"Transcription: <|audio|><|endofaudio|> {feature['text']}" for feature in features]
        text_processed = self.tokenizer(
            text_inputs,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        # Create labels (shifted input_ids)
        labels = text_processed.input_ids.clone()
        labels[labels == self.tokenizer.pad_token_id] = -100  # Ignore padding in loss
        
        return {
            "audio_input": input_features,
            "audio_attention_mask": audio_attention_mask,
            "input_ids": text_processed.input_ids,
            "attention_mask": text_processed.attention_mask,
            "labels": labels
        }

from transformers import TrainerCallback

class SaveModelCallback(TrainerCallback):
    def on_save(self, args, state, control, model, **kwargs):
        if state.is_world_process_zero:
            # Save using torch.save instead of safetensors
            output_dir = f"{args.output_dir}/checkpoint-{state.global_step}"
            model.save_pretrained(
                output_dir,
                is_main_process=True,
                save_function=torch.save,
                safe_serialization=False,
            )
            model.tokenizer.save_pretrained(output_dir)
            model.processor.save_pretrained(output_dir)
            print(f"Saved model checkpoint to {output_dir}")
        return control
    

def save_model_with_shared_weights(model, output_dir):
    """Save model with proper handling of shared weights"""
    import os
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Save the language model
    lm_dir = os.path.join(output_dir, "lm")
    os.makedirs(lm_dir, exist_ok=True)
    model.lm.save_pretrained(
        lm_dir,
        is_main_process=True,
        save_function=torch.save,
        safe_serialization=False,  # Disable safetensors
    )
    
    # 2. Save the audio encoder
    encoder_dir = os.path.join(output_dir, "audio_encoder")
    os.makedirs(encoder_dir, exist_ok=True)
    model.audio_encoder.save_pretrained(
        encoder_dir,
        is_main_process=True,
        save_function=torch.save,
        safe_serialization=False,
    )
    
    # 3. Save the projection layer weights separately
    projection_path = os.path.join(output_dir, "projection_layer.pt")
    torch.save(model.audio_projection_layer.state_dict(), projection_path)
    
    # 4. Save tokenizer and processor
    model.tokenizer.save_pretrained(output_dir)
    model.processor.save_pretrained(output_dir)
    
    # 5. Save configuration
    config = {
        "lm_id": model.lm.config._name_or_path,
        "audio_encoder_id": model.audio_encoder.config._name_or_path,
        "audio_dim": model.audio_dim,
        "lm_dim": model.lm_dim,
        "use_pooler": model.use_pooler,
        "special_tokens": ["<|audio|>", "<|endofaudio|>"] + [f"<|{id}|>" for id in range(1, 27)]
    }
    
    # Save config as JSON
    import json
    with open(os.path.join(output_dir, "model_config.json"), "w") as f:
        json.dump(config, f)
    
    print(f"Model successfully saved to {output_dir}")
# Example usage
def main():
    path_model = 'Qwen/Qwen2.5-0.5B-Instruct'
    audio_encoder = 'vinai/PhoWhisper-large'
    # Initialize model
    model = SpeechToTextModel(
        audio_encoder_id=audio_encoder,
        lm_id=path_model,
        freeze_encoder=True,  # For initial training, freeze encoder
        freeze_lm=False       # Fine-tune LM
    )
    model = model.to(dtype = torch.bfloat16)
    # Create dataset
    # This is just a placeholder - you would need to create an actual dataset
    from datasets import load_dataset
    
    # Load CommonVoice dataset as an example
    dataset = load_dataset("Colder203/Audio_Robot_Interaction", split="train")
    
    # Preprocess dataset
    def preprocess_function(examples):
        audio = examples["audio"]
        text = examples["english"]
        return {"audio": audio["array"], "text": text}
    
    processed_dataset = dataset.map(preprocess_function)
    
    # Create data collator
    data_collator = SpeechToTextDataCollator(
        processor=model.processor, 
        tokenizer=model.tokenizer
    )
    
    # Training arguments
    training_args = TrainingArguments(
        logging_steps=2,
        output_dir="./speech_to_text_model",
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=5e-5,
        num_train_epochs=1,
        save_strategy="epoch",
        # evaluation_strategy="epoch",
        logging_dir="./audio/logs",
        remove_unused_columns=False,
        report_to = "tensorboard",
        bf16 = True,
        save_safetensors=False,
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=processed_dataset,
        data_collator=data_collator,
        # callbacks=[SaveModelCallback()]
    )
    
    # Train model
    trainer.train()
    
    # Save model
    # model.save_pretrained("./speech_to_text_model_final")
    save_model_with_shared_weights(model, "./speech_to_text_model_final")
    
    # Example inference
    # Load audio file
    print('Save successful')
    audio_file = "audio/303.wav"
    from datasets import Audio
    audio = Audio().decode_example({"path": audio_file})
    
    audio_input = model.processor(
        audio=audio["array"], 
        sampling_rate=16000, 
        return_tensors="pt"
    ).input_features.to(model.device)
    
    # Generate text
    transcription = model.generate_from_audio(
        audio_input=audio_input,
        max_length=100,
        num_beams=4,
        temperature=0.7
    )
    
    print(f"Transcription: {transcription[0]}")


if __name__ == "__main__":
    main()