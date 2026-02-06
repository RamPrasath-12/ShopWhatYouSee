# Helper script to write local_llm_efficient.py
# Run this once to create the file

code = '''"""Local LLM - Phi-3 Mini"""
import re, json, logging
from pathlib import Path
from .prompts import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE

MODEL_DIR = Path(__file__).parent.parent / 'data' / 'llm'
GGUF_MODEL_PATH = MODEL_DIR / 'phi3-finetuned-q4.gguf'

_instance = None

def get_llm():
    global _instance
    if _instance is None:
        _instance = LocalLLMEfficient()
    return _instance

def preload_llm():
    get_llm().load()

class LocalLLMEfficient:
    def __init__(self):
        self._llm = None
        self._loaded = False
    
    def load(self):
        if self._loaded: return True
        try:
            from llama_cpp import Llama
            if not GGUF_MODEL_PATH.exists():
                print('[LocalLLM] GGUF not found')
                return False
            print('[LocalLLM] Loading...')
            self._llm = Llama(model_path=str(GGUF_MODEL_PATH), n_ctx=2048, n_threads=6, n_gpu_layers=0, verbose=False)
            self._loaded = True
            print('[LocalLLM] Loaded!')
            return True
        except Exception as e:
            print(f'[LocalLLM] Failed: {e}')
            return False
    
    def generate_filters(self, category, attributes, scene, query, session_history=None):
        if not self._loaded and not self.load():
            return self._fallback_response(query, attributes)
        
        attr_str = json.dumps(attributes) if attributes else '{}'
        user_content = USER_PROMPT_TEMPLATE.format(
            category=category, agman_attributes=attr_str, 
            scene=scene or 'unknown', user_query=query
        )
        if session_history:
            user_content += chr(10) + 'Previous: ' + ', '.join(session_history[-3:])
        
        # Phi-3 tags
        SYS = chr(60) + '|system|' + chr(62)
        END = chr(60) + '|end|' + chr(62)
        USR = chr(60) + '|user|' + chr(62)
        AST = chr(60) + '|assistant|' + chr(62)
        NL = chr(10)
        
        full_prompt = SYS + NL + SYSTEM_PROMPT + NL + END + NL
        full_prompt += USR + NL + user_content + NL + END + NL
        full_prompt += AST + NL
        
        try:
            output = self._llm(full_prompt, max_tokens=400, temperature=0.1, stop=[END, 'User:'])
            text = output['choices'][0]['text'].strip()
            return self._parse_response(text, attributes)
        except Exception as e:
            print(f'[LocalLLM] Generation failed: {e}')
            return self._fallback_response(query, attributes)
    
    def _parse_response(self, text, attributes):
        try:
            match = re.search(r'{.*}', text, re.DOTALL)
            if match:
                data = json.loads(match.group())
                filters = data.get('filters', {})
                # Include AGMAN attributes if not overridden
                if attributes:
                    if attributes.get('color_hex') and 'color' not in filters:
                        filters['color'] = attributes['color_hex']
                    if attributes.get('pattern') and 'pattern' not in filters:
                        filters['pattern'] = attributes['pattern']
                    if attributes.get('sleeve') and 'sleeve' not in filters:
                        filters['sleeve'] = attributes['sleeve']
                return {'reasoning': data.get('reasoning', ''), 'filters': filters, 'confidence': data.get('confidence', 0.7)}
        except Exception as e:
            print(f'[LocalLLM] Parse failed: {e}')
        return self._fallback_response('', attributes)
    
    def _fallback_response(self, query, attributes):
        filters = {}
        if attributes:
            if attributes.get('color_hex'): filters['color'] = attributes['color_hex']
            if attributes.get('pattern'): filters['pattern'] = attributes['pattern']
            if attributes.get('sleeve'): filters['sleeve'] = attributes['sleeve']
        q = query.lower() if query else ''
        for c in ['red','blue','green','black','white','pink']:
            if c in q:
                filters['color'] = c
                break
        return {'reasoning': 'Fallback', 'filters': filters, 'confidence': 0.3}
'''

with open('models/local_llm_efficient.py', 'w', encoding='utf-8') as f:
    f.write(code)
print('File written successfully!')
