#!/usr/bin/env python3
"""
mistral_client.py
-----------------
Client minimaliste et réutilisable pour le chatbot Mistral AI.

Usage simple :

    from mistral_client import MistralChat

    history = []                                 # conversation (optionnel)
    chat = MistralChat()                         # lit $MISTRAL_API_KEY
    answer = chat.send(prompt="Bonjour !",       # prompt utilisateur
                       messages=history)         # historique en mémoire
    print(answer)

• Dépendance : requests  (pip install requests)
• Authentification :
      ─ Variable d’environnement MISTRAL_API_KEY
      ─ ou paramètre api_key="mistral-xxx" passé au constructeur
"""

from __future__ import annotations

import os
import time
import json
from typing import List, Dict, Optional

import requests


from dotenv import load_dotenv

load_dotenv(dotenv_path="app/.env")


class MistralChat:
    """Enveloppe ultra-légère pour l’endpoint /chat/completions de Mistral AI."""

    _API_URL = "https://api.mistral.ai/v1/chat/completions"
    _DEFAULT_MODEL = "mistral-small-latest"

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = _DEFAULT_MODEL,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        timeout: int = 60,
    ) -> None:
        self.api_key = os.getenv("MISTRAL_API_KEY")
        
        if not self.api_key:
            raise ValueError(
                "Aucune clé API trouvée : définissez-la via MISTRAL_API_KEY "
                "ou passez-la au constructeur."
            )

        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout

        self._headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    # ------------------------------------------------------------------ #
    #  Méthode publique principale
    # ------------------------------------------------------------------ #
    def send(
        self,
        prompt: str,
        messages: Optional[List[Dict[str, str]]] = None,
    ) -> str:
        """
        Envoie `prompt` + `messages` à l’API et renvoie la réponse.

        Paramètres
        ----------
        prompt : str
            Message utilisateur courant.
        messages : list[dict] | None
            Historique au format OpenAI-style :
            [{"role": "user"|"assistant"|"system", "content": "..."}]

        Retour
        ------
        str : contenu renvoyé par l’assistant.
        """
        # On part d’une copie pour ne pas muter la liste d’appel
        thread: List[Dict[str, str]] = list(messages or [])
        thread.append({"role": "user", "content": prompt})

        while True:
            try:
                resp = requests.post(
                    self._API_URL,
                    headers=self._headers,
                    json={
                        "model": self.model,
                        "messages": thread,
                        "temperature": self.temperature,
                        "max_tokens": self.max_tokens,
                    },
                    timeout=self.timeout,
                )

                # ---------- quotas free-tier -------------------------------- #
                if resp.status_code == 429:  # rate-limit
                    retry = int(resp.headers.get("Retry-After", "5"))
                    print(f"⏳  Limite atteinte, nouvel essai dans {retry}s …")
                    time.sleep(retry)
                    continue

                resp.raise_for_status()  # lève HTTPError si 4xx/5xx
                data = resp.json()
                answer = data["choices"][0]["message"]["content"].strip()
                return answer

            except requests.HTTPError as http_err:
                if resp.status_code == 401:
                    raise RuntimeError("Clé API invalide ou expirée.") from http_err
                raise

            except requests.RequestException as net_err:
                raise RuntimeError(f"Erreur réseau : {net_err}") from net_err

            except (KeyError, IndexError, json.JSONDecodeError) as parse_err:
                raise RuntimeError(
                    f"Réponse JSON inattendue : {parse_err}"
                ) from parse_err


# ---------------------------------------------------------------------- #
#  Exécution directe en console (optionnelle)
# ---------------------------------------------------------------------- #
if __name__ == "__main__":
    print("🗨️  Mode interactif — tapez 'quitter' pour sortir.")
    history: List[Dict[str, str]] = []
    chat = MistralChat()  # clé lue dans l’environnement

    while True:
        user_text = input("> ").strip()
        if user_text.lower() == "quitter":
            print("Au revoir !")
            break
        try:
            reply = chat.send(prompt=user_text, messages=history)
            history.append({"role": "user", "content": user_text})
            history.append({"role": "assistant", "content": reply})
            print(reply)
        except Exception as exc:
            print(f"⚠️  {exc}")





"""
# autre_script.py
from mistral_client import MistralChat

conversation = [
    {"role": "system", "content": "Tu es un assistant très bref."}
]

chat = MistralChat()           # api_key lue dans $MISTRAL_API_KEY
réponse = chat.send(
    prompt="Comment vas-tu ?",
    messages=conversation
)
print(réponse)
conversation.append({"role": "assistant", "content": réponse})

"""