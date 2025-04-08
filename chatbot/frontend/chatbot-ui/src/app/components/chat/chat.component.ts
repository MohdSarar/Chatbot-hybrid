/*
  chat.component.ts

  Composant principal de chat :
    1) Affiche un résumé du profil + un bouton "Modifier" 
       ou bien le formulaire, selon showForm().
    2) Stocke l’historique des messages (signal<ChatMessage[]>).
    3) Fait appel aux endpoints /recommend et /query pour récupérer
       des réponses de l’assistant (messages assistant).
    4) Gère le scroll automatique vers le bas du chat-history,
       en observant tout changement de messages() (signal).
*/

import { Component, OnInit, ViewChild, ElementRef, AfterViewChecked, signal, effect } from '@angular/core';
import { HttpClient, HttpClientModule } from '@angular/common/http';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';

// Composants enfants
import { UserProfileFormComponent } from '../user-profile-form/user-profile-form.component';
import { MessageBubbleComponent } from '../message-bubble/message-bubble.component';

// Modèles
import { UserProfile } from '../../models/user-profile.model';
import { ChatMessage } from './../../models/chat.models';

@Component({
  selector: 'app-chat',
  standalone: true,
  imports: [
    CommonModule,
    FormsModule,
    HttpClientModule,
    UserProfileFormComponent,
    MessageBubbleComponent
  ],
  templateUrl: './chat.component.html',
  styleUrls: ['./chat.component.scss']
})
export class ChatComponent implements OnInit, AfterViewChecked {

  // Profil utilisateur => null si pas défini
  profile = signal<UserProfile | null>(null);

  // Historique de messages
  messages = signal<ChatMessage[]>([]);

  // Indicateur d’envoi (disable input/bouton)
  sending = signal(false);

  // Saisie courante du champ input
  currentInput = signal('');

  // showForm => True => affiche le formulaire
  //          => False => affiche résumé du profil
  showForm = signal(true);

  // Conteneur DOM de l’historique
  @ViewChild('chatHistoryContainer')
  chatHistoryContainer!: ElementRef<HTMLDivElement>;

  // Endpoint backend
  private API_BASE_URL = 'http://localhost:8000';

  constructor(private http: HttpClient) {
    // Effet sur messages() => scroll auto
    effect(() => {
      const _ = this.messages(); // lecture du signal
      this.scrollToBottom();
    });
  }

  ngOnInit(): void {
    // Rien de spécial
  }

  ngAfterViewChecked() {
    // Scroll auto si un profil est défini
    if (this.profile()) {
      this.scrollToBottom();
    }
  }

  toggleForm(): void {
    this.showForm.set(!this.showForm());
  }

  // Lors de la soumission du formulaire
  onProfileSubmit(profileData: UserProfile): void {
    this.profile.set(profileData);
    this.showForm.set(false);
    this.getRecommendation();
  }

  // Appel /recommend
  private getRecommendation(): void {
    const p = this.profile();
    if (!p) return;
    this.sending.set(true);

    this.http.post<{ recommended_course: string; reply: string; details?: any }>(
      `${this.API_BASE_URL}/recommend`,
      { profile: p }
    ).subscribe({
      next: (response) => {
        this.sending.set(false);
        if (response) {
          // Met à jour le profil avec la formation recommandée
          const updated = {
            ...p,
            recommended_course: response.recommended_course
          } as UserProfile;
          this.profile.set(updated);

          // Construit un message assistant structuré
          const formattedReply: ChatMessage = {
            role: 'assistant',
            content: JSON.stringify({
              reply: response.reply,
              course: response.recommended_course,
              details: response.details
            })
          };
          // Ajoute ce msg dans l’historique
          this.messages.update(msgs => [...msgs, formattedReply]);
        }
      },
      error: () => {
        this.sending.set(false);
        this.messages.update(msgs => [
          ...msgs,
          { role: 'assistant', content: 'Erreur lors de la recommandation.' }
        ]);
      }
    });
  }

  // Envoi d'un message user => /query
  sendMessage(): void {
    const p = this.profile();
    if (!p) return;
    const inputText = this.currentInput().trim();
    if (!inputText) return;

    // Ajout local d'un message user
    this.messages.update(msgs => [
      ...msgs,
      { role: 'user', content: inputText }
    ]);

    const payload = {
      profile: p,
      history: this.messages().slice(0, -1),
      question: inputText
    };

    this.sending.set(true);
    this.http.post<{ reply: string }>(
      `${this.API_BASE_URL}/query`,
      payload
    ).subscribe({
      next: (res) => {
        this.sending.set(false);
        if (res?.reply) {
          // Ajout d'un message assistant
          this.messages.update(msgs => [
            ...msgs,
            { role: 'assistant', content: res.reply }
          ]);
        }
        // Réinit champ input
        this.currentInput.set('');
      },
      error: () => {
        this.sending.set(false);
        this.messages.update(msgs => [
          ...msgs,
          { role: 'assistant', content: 'Erreur lors de la communication.' }
        ]);
      }
    });
  }

  // Scroll auto
  private scrollToBottom(): void {
    try {
      if (!this.chatHistoryContainer?.nativeElement) return;
      const container = this.chatHistoryContainer.nativeElement;
      container.scrollTop = container.scrollHeight;
    } catch (e) {
      console.error('[Scroll error]', e);
    }
  }

  // Envoi d’un récap par e-mail => /send-email
  sendRecapEmail(): void {
    const p = this.profile();
    if (!p) return;

    // Vérifie la présence d'un email
    if (!p.email) {
      this.messages.update(msgs => [
        ...msgs,
        { role: 'assistant', content: "Veuillez renseigner un email avant l’envoi du récap." }
      ]);
      return;
    }

    // On doit envoyer { profile, chatHistory } au backend
    // car /send-email attend un SendEmailRequest
    const payload = {
      profile: p,
      chatHistory: this.messages() // ex: [ { role, content }, ... ]
    };

    this.http.post<{ status: string }>(
      `${this.API_BASE_URL}/send-email`,
      payload
    ).subscribe({
      next: (res) => {
        this.messages.update(msgs => [
          ...msgs,
          { role: 'assistant', content: `Status: ${res.status}. Un récap vous a été envoyé.` }
        ]);
      },
      error: () => {
        this.messages.update(msgs => [
          ...msgs,
          { role: 'assistant', content: 'Erreur lors de l’envoi de l’email.' }
        ]);
      }
    });
  }
}




// 16/04 => 9h 65€
// OPEL - Claro Automobiles Chartres