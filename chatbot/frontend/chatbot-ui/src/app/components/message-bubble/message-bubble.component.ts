/*
  message-bubble.component.ts
  
  On applique directement la classe 'assistant-msg' ou 'user-msg'
  sur le conteneur principal .message-bubble
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
  @Input() message!: ChatMessage;

  isUser(): boolean {
    return this.message.role === 'user';
  }

  isAssistant(): boolean {
    return this.message.role === 'assistant';
  }

  /**
   * Teste si c'est un message structuré (recommandation).
   * On vérifie la présence de .reply et .course
   */
  isStructured(): boolean {
    try {
      const parsed = JSON.parse(this.message.content);
      return parsed?.reply && parsed?.course;
    } catch {
      return false;
    }
  }

  /** Parse le content JSON */
  parse(): any {
    try {
      return JSON.parse(this.message.content);
    } catch {
      return null;
    }
  }

  /** Classe user-msg ou assistant-msg en fonction du rôle */
  getBubbleClasses(): string {
    return this.isUser() ? 'user-msg' : 'assistant-msg';
  }

  /** Avatar en fonction du rôle */
  getAvatarPath(): string {
    return this.isAssistant() ? 'assets/ai.png' : 'assets/user.png';
  }
}
