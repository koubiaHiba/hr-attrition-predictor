# run_with_ngrok.py
import os
from pyngrok import ngrok
import subprocess
import time
import webbrowser
from dotenv import load_dotenv

# Charger le token depuis .env
load_dotenv()
NGROK_AUTH_TOKEN = os.getenv("3FKavVlZ90kjZ9kvUclfDS9dUun_2f95y4MmRvDmKTiA1h5qK")

STREAMLIT_PORT = 8501

def main():
    print("="*60)
    print("🚀 LANCEUR STREAMLIT + NGROK")
    print("="*60)
    
    # Vérifier le token
    if not NGROK_AUTH_TOKEN or NGROK_AUTH_TOKEN == "TON_TOKEN_NGROK_ICI":
        print("❌ Token ngrok non configuré !")
        print("📋 Créez un fichier .env avec : NGROK_AUTH_TOKEN=votre_token")
        print("🔑 Obtenez votre token sur : https://dashboard.ngrok.com/")
        return
    
    try:
        print("🔧 Configuration de ngrok...")
        ngrok.set_auth_token(NGROK_AUTH_TOKEN)
        
        print("🔗 Création du tunnel ngrok...")
        public_url = ngrok.connect(STREAMLIT_PORT, "http")
        
        print("\n" + "="*60)
        print("✅ APPLICATION DISPONIBLE !")
        print("="*60)
        print(f"🔗 URL publique : {public_url}")
        print("📊 Dashboard ngrok : http://localhost:4040")
        print("="*60)
        
        webbrowser.open(public_url)
        print("🌐 Ouverture dans le navigateur...")
        
        print("\n🚀 Lancement de Streamlit...")
        cmd = [
            "streamlit", "run", "app.py",
            "--server.port", str(STREAMLIT_PORT),
            "--server.headless", "true"
        ]
        
        process = subprocess.Popen(cmd)
        
        print("\n✅ Application lancée !")
        print("💡 Appuyez sur Ctrl+C pour arrêter\n")
        
        process.wait()
                
    except Exception as e:
        print(f"❌ Erreur : {e}")
        
    finally:
        print("\n🛑 Arrêt en cours...")
        ngrok.kill()

if __name__ == "__main__":
    main()