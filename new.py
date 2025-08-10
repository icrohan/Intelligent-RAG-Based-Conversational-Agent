import speech_recognition as sr
import pyttsx3
import numpy as np
import time
import random
import audioop
import threading
import queue
import torch
from huggingface_hub import InferenceClient
from sentence_transformers import SentenceTransformer

class VoiceActivityDetector:
    def __init__(self, energy_threshold=300, silence_threshold=30):
        self.energy_threshold = energy_threshold
        self.silence_threshold = silence_threshold
        
    def is_speech(self, audio_data, sample_width):
        rms = audioop.rms(audio_data, sample_width)
        return rms > self.energy_threshold

class InteractiveBot:
    def __init__(self):
        # Initialize speech recognition and text-to-speech
        self.recognizer = sr.Recognizer()
        self.engine = pyttsx3.init()
        self.vad = VoiceActivityDetector()
        
        # Configure voice
        voices = self.engine.getProperty('voices')
        self.engine.setProperty('rate', 120)
        self.engine.setProperty('voice', voices[1].id)
        
        # Load models
        self.load_models()
        
        # State tracking
        self.is_awake = False
        self.last_activity = time.time()
        self.is_speaking = False
        self.should_stop_speaking = False
        self.speech_queue = queue.Queue()
        
    def load_models(self):
        """Load and prepare models"""
        print("Loading models and documents...")

        # Load document for question-answering
        with open(r"testing_doc.txt", 'r', encoding='utf-8') as file:
            self.data = file.read().split('\n\n')
            
        self.sent_transformer = SentenceTransformer('sentence-transformers/all-MiniLM-l6-v2')
        self.encodings_to_chunks = {}

        for chunk in self.data:
            encoded_chunk = self.sent_transformer.encode(chunk)
            self.encodings_to_chunks[tuple(encoded_chunk)] = chunk
            
        self.client = InferenceClient(api_key="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")

        # Load classification model
        self.classifier = general_classifier(6)
        self.encoder = SentenceTransformer("sentence-transformers/all-MiniLM-l6-v2")
        self.classifier.load_state_dict(torch.load("classifier.pt"))
        self.classifier.eval()

        print("System ready!")

    def classify_command(self, text):
        """Classify input text into predefined categories"""
        encoded = self.encoder.encode(text)
        encoded = np.expand_dims(encoded, axis=0)
        x = torch.tensor(encoded, dtype=torch.float32)
        model_out = self.classifier(x)
        idx = idx_to_class[model_out.argmax().item()]
        return idx

    def process_question(self, question):
        """Process and answer a general question"""
        ques_encoded = np.array(self.sent_transformer.encode([question])).squeeze(axis=0).transpose()
        chunk_score_list = []

        for embeddings, chunk in self.encodings_to_chunks.items():
            sim_score = np.dot(ques_encoded, embeddings) / (np.linalg.norm(ques_encoded) * np.linalg.norm(embeddings))
            chunk_score_list.append([chunk, sim_score])

        chunk_score_list.sort(key=lambda a: a[-1], reverse=True)

        # Generate context
        context = ''
        for i in range(2):
            for x in chunk_score_list[i][0].split('.'):
                context += x

        # Get answer from model
        instruction = f'CONTEXT: {context}\nQUESTION: {question}\nAnswer the question based on the context provided in 50 words.'
        messages = [{"role": "user", "content": instruction}]

        stream = self.client.chat.completions.create(
            model="Qwen/Qwen2.5-72B-Instruct",
            messages=messages,
            temperature=0.7,
            max_tokens=256,
            top_p=0.7,
            stream=True
        )

        return ''.join(chunk.choices[0].delta.content for chunk in stream)

    def run(self):
        """Main loop"""
        self.speak("System initialized. Say Hello Bot to start.")

        while True:
            try:
                # Listen for voice activity
                text = self.listen_for_voice_activity()

                if text:
                    self.last_activity = time.time()

                    # Wake word detection
                    if 'hello' in text and not self.is_awake:
                        self.is_awake = True
                        self.speak("Hello! How can I help you?")
                        continue

                    if self.is_awake:
                        if 'goodbye' in text or 'bye' in text:
                            self.speak("Goodbye! Have a great day!")
                            self.is_awake = False
                            continue

                        # Determine if it's a command or a general question
                        command_category = self.classify_command(text)
                        if command_category in idx_to_class.values():
                            self.speak(f"Executing command: {command_category}")
                        else:
                            self.speak("Let me think about that...")
                            answer = self.process_question(text)
                            self.speak(answer)
                            self.speak("What else would you like to know?")

            except Exception as e:
                print(f"Error: {e}")
                if self.is_awake:
                    self.speak("I encountered an error. Please try again.")
                continue

if __name__ == "__main__":
    bot = InteractiveBot()
    bot.run()
