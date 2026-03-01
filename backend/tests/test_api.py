import google.generativeai as genai

genai.configure(api_key="AIzaSyBvHydmgRp04qfflrzOvMNiJYtfIwy_fDA")


# models = genai.list_models()
# for m in models:
#     print(m.name, "→ supports:", m.supported_generation_methods)
    
    
    
model = genai.GenerativeModel("models/gemini-flash-latest")

resp = model.generate_content("Say OK in JSON")
print(resp.text)
   
