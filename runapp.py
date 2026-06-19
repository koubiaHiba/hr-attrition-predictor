import streamlit.web.cli as stcli
import sys
import os
from pyngrok import ngrok
import threading
import time
import subprocess

# === CONFIGURATION ===
NGROK_AUTH_TOKEN = "3FKavVlZ90kjZ9kvUclfDS9dUun_2f95y4MmRvDmKTiA1h5qK"
STREAMLIT_PORT = 8501

# === ÉTAPE 1 : Configurer ngrok ===
ngrok.set_auth_token(NGROK_AUTH_TOKEN)

# === ÉTAPE 2 : Démarrer le tunnel ngrok ===
public_url = ngrok.connect(STREAMLIT_PORT, "http")
print(f"\n✅ Tunnel ngrok créé avec succès !")
print(f"🔗 URL publique : {public_url}")
print(f"📊 Tableau de bord ngrok : http://localhost:4040\n")

# === ÉTAPE 3 : Lancer l'application Streamlit ===
print("🚀 Lancement de l'application Streamlit...")
print("⏳ Patientez quelques secondes...")

# Lancer Streamlit en tant que sous-processus
cmd = ["streamlit", "run", "app.py", "--server.port", str(STREAMLIT_PORT)]
process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

# Attendre que le serveur soit prêt
time.sleep(3)

print("\n" + "="*60)
print("✅ APPLICATION DISPONIBLE !")
print("="*60)
print(f"🌐 URL publique : {public_url}")
print("="*60)
print("\n💡 Pour arrêter : Ctrl+C")

# Garder le script en cours d'exécution
try:
    process.wait()
except KeyboardInterrupt:
    print("\n🛑 Arrêt de l'application...")
    process.terminate()
    ngrok.kill()