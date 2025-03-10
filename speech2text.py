import torch
import torch.nn as nn
from transformers import Wav2Vec2Model, Wav2Vec2Processor, AutoTokenizer, AutoModelForCausalLM, AutoProcessor
from transformers import WhisperModel
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions
from transformers.modeling_utils import PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions
from transformers.modeling_utils import SequenceSummary

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
    # @classmethod
    # def from_pretrained(cls, model_dir):
    #     """Load model from directory"""
    #     import os
    #     import json
        
    #     # Load config
    #     with open(os.path.join(model_dir, "model_config.json"), "r") as f:
    #         config = json.load(f)
        
    #     # Initialize model with saved components
        
    #     model = cls(
    #         audio_encoder_id=os.path.join(model_dir, "audio_encoder"),
    #         lm_id=os.path.join(model_dir, "lm"),
    #         use_pooler=config["use_pooler"]
    #     )
        
    #     # Load projection layer weights
    #     projection_path = os.path.join(model_dir, "projection_layer.pt")
    #     model.audio_projection_layer.load_state_dict(torch.load(projection_path))
        
    #     return model
    @classmethod
    def from_pretrained(cls, model_dir):
        """Load model from directory"""
        import os
        import json
        import torch
        from transformers import WhisperConfig, WhisperModel, AutoModelForCausalLM
        
        # Load config
        with open(os.path.join(model_dir, "model_config.json"), "r") as f:
            config = json.load(f)
        
        # Load the saved encoder config directly instead of creating a new instance
        encoder_dir = os.path.join(model_dir, "audio_encoder")
        encoder_config_path = os.path.join(encoder_dir, "config.json")
        with open(encoder_config_path, "r") as f:
            encoder_config_dict = json.load(f)
        
        # Create encoder config object and then model with exact same architecture
        if "whisper" in encoder_config_dict["_name_or_path"].lower():
            encoder_config = WhisperConfig.from_dict(encoder_config_dict)
            audio_encoder = WhisperModel(encoder_config)
            # Load saved weights
            audio_encoder.load_state_dict(torch.load(os.path.join(encoder_dir, "pytorch_model.bin")), strict=False)
        else:
            # For other encoder types
            audio_encoder = Wav2Vec2Model.from_pretrained(encoder_dir)
        
        # Load LM
        lm_dir = os.path.join(model_dir, "lm")
        lm = AutoModelForCausalLM.from_pretrained(lm_dir)
        
        # Create processor and tokenizer
        processor = AutoProcessor.from_pretrained(model_dir)
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        
        # Create model instance without loading the weights
        # We'll manually set the components
        model = cls.__new__(cls)
        nn.Module.__init__(model)
        
        # Set model components
        model.audio_encoder = audio_encoder
        model.lm = lm
        model.processor = processor
        model.tokenizer = tokenizer
        model.use_pooler = config["use_pooler"]
        model.audio_dim = config["audio_dim"]
        model.lm_dim = config["lm_dim"]
        model.audio_token_id = model.tokenizer.convert_tokens_to_ids("<|audio|>")
        model.audio_end_token_id = model.tokenizer.convert_tokens_to_ids("<|endofaudio|>")
        
        # Initialize the pooler
        model.audio_avg_pooler = nn.AvgPool1d(kernel_size=2, stride=2, padding=0)
        
        # Load projection layer weights
        model.audio_projection_layer = MultiModalProjector(
            in_dim=model.audio_dim, 
            out_dim=model.lm_dim
        )
        projection_path = os.path.join(model_dir, "projection_layer.pt")
        model.audio_projection_layer.load_state_dict(torch.load(projection_path))
        
        return model