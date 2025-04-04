/*
  chat.component.ts
  Composant principal de chat :
    1) Affiche un résumé du profil + "Modifier"
       ou le formulaire si l'utilisateur le souhaite.
    2) Stocke l’historique de messages,
       et scrolle vers le haut (flex-direction: column-reverse).
    3) Interroge /recommend et /query pour avoir un retour minimal.
*/

import { Component, OnInit, ViewChild, ElementRef, AfterViewChecked, signal, effect } from '@angular/core';
import { HttpClient, HttpClientModule } from '@angular/common/http';
import { CommonModule } from '@angular/common';

import { FormsModule } from '@angular/forms';

import { UserProfileFormComponent } from '../user-profile-form/user-profile-form.component';
import { MessageBubbleComponent } from '../message-bubble/message-bubble.component';

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
export class ChatComponent implements OnInit {

  // Profil de l’utilisateur. Null si non encore défini.
  profile = signal<UserProfile | null>(null);

  // Historique des messages
  messages = signal<ChatMessage[]>([]);

  // Indicateur d’envoi (pour désactiver l’input)
  sending = signal(false);

  // Saisie utilisateur dans le chat
  currentInput = signal('');

  /*
    showForm : True => le formulaire s’affiche
               False => on affiche le résumé du profil + bouton "Modifier"
  */
  showForm = signal(true);

  // Référence au conteneur DOM de l’historique
  @ViewChild('chatHistoryContainer') chatHistoryContainer!: ElementRef;

  private API_BASE_URL = 'http://localhost:8000';

  constructor(private http: HttpClient) {
    // Surveille toute mise à jour de messages()
    effect(() => {
      const _ = this.messages();  // lecture du signal
      this.scrollToBottom();      // se positionne tout en bas
    });
  }

  ngOnInit(): void {
    // Rien de particulier
  }

  ngAfterViewChecked(): void {
    this.scrollToBottom();
  }

  /*
    Bascule entre affichage du formulaire et affichage du résumé.
  */
  toggleForm(): void {
    this.showForm.set(!this.showForm());
  }

  /*
    Quand l’utilisateur soumet le profil dans le composant enfant.
    1) Stocke le profil
    2) Cache le formulaire
    3) Lance la recommandation
  */
  onProfileSubmit(profileData: UserProfile): void {
    this.profile.set(profileData);
    this.showForm.set(false);
    this.getRecommendation();
  }

  /*
    Requête /recommend. Ajoute ensuite un message assistant dans l’historique.
  */
    private getRecommendation(): void {
      if (!this.profile()) return;
    
      this.sending.set(true);
    
      this.http.post<{ recommended_course: string; reply: string; details?: any }>(
        `${this.API_BASE_URL}/recommend`,
        { profile: this.profile() }
      ).subscribe({
        next: (response) => {
          this.sending.set(false);
          if (response) {
            const updatedProfile = {
              ...this.profile(),
              recommended_course: response.recommended_course
            } as UserProfile;
            this.profile.set(updatedProfile);
    
            // Structuration du message avec les détails séparés
            const formattedReply: ChatMessage = {
              role: 'assistant',
              content: JSON.stringify({
                reply: response.reply,
                course: response.recommended_course,
                details: response.details
              })
            };
    
            this.messages.update(msgs => [...msgs, formattedReply]);
          }
        },
        error: () => {
          this.sending.set(false);
          this.messages.update(msgs => [
            ...msgs,
            { role: 'assistant', content: "Erreur lors de la recommandation." }
          ]);
        }
      });
    }
    

  /*
    Envoi d’un message utilisateur à /query, ajout d’une réponse assistant.
  */
  sendMessage(): void {
    const p = this.profile();
    if (!p) return;

    const inputText = this.currentInput().trim();
    if (!inputText) return;

    // Ajout du message user
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
        if (res && res.reply) {
          this.messages.update(msgs => [
            ...msgs,
            { role: 'assistant', content: res.reply }
          ]);
        }
        // Vide le champ
        this.currentInput.set('');
      },
      error: () => {
        this.sending.set(false);
        this.messages.update(msgs => [
          ...msgs,
          { role: 'assistant', content: "Erreur lors de la communication." }
        ]);
      }
    });
  }

  /*
    scrollToBottom:
    Met l’ascenseur du conteneur au niveau de scrollHeight,
    de sorte que le dernier message soit toujours visible.
  */
    private scrollToBottom(): void {
    try {
      const container = this.chatHistoryContainer?.nativeElement;
      container.scrollTop = container.scrollHeight;
    } catch (err) {
      console.warn("Scroll impossible :", err);
    }
  }
}
