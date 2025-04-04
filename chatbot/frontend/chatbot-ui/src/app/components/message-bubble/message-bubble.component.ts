/*
  Composant MessageBubbleComponent
  Gère l'affichage conditionnel de chaque message dans le chat :
  - message texte classique
  - message structuré avec formation recommandée
  - avatar selon le rôle (user ou assistant)
*/

import { Component, Input } from '@angular/core';
import { CommonModule } from '@angular/common';
import { ChatMessage } from '../../models/chat.models';

@Component({
  selector: 'app-message-bubble',
  standalone: true,
  imports: [CommonModule],
  templateUrl: './message-bubble.component.html',
  styleUrls: ['./message-bubble.component.scss']
})
export class MessageBubbleComponent {
  // Message à afficher (texte ou JSON structuré)
  @Input() message!: ChatMessage;

  /*
   * Détermine si le contenu est une réponse structurée
   * On teste si c'est un objet JSON contenant "reply" et "course"
   */
  isStructured(): boolean {
    try {
      const parsed = JSON.parse(this.message.content);
      return parsed?.reply && parsed?.course;
    } catch {
      return false;
    }
  }

  /*
   * Parse le message.content (JSON string) en objet
   */
  parse(): any {
    try {
      return JSON.parse(this.message.content);
    } catch {
      return null;
    }
  }

  /*
   * Retourne le chemin de l’avatar à utiliser :
   * - assistant => assets/ai.png
   * - user      => assets/user.png
   */
  getAvatarPath(): string {
    return this.message.role === 'assistant'
      ? 'assets/ai.png'
      : 'assets/user.png';
  }
}
