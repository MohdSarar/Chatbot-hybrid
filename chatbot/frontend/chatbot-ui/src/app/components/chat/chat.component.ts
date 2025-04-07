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
    HttpClientModule, // <-- si tu veux injecter HttpClient, importer HttpClientModule
    UserProfileFormComponent,
    MessageBubbleComponent
  ],
  templateUrl: './chat.component.html',
  styleUrls: ['./chat.component.scss']
})
export class ChatComponent implements OnInit, AfterViewChecked {

  /*
   * Profil utilisateur => null si pas encore renseigné.
   * Quand le formulaire est soumis, on met un UserProfile dedans.
   */
  profile = signal<UserProfile | null>(null);

  /*
   * messages() : Historique de messages du chat
   * ( ex: [ { role: 'assistant', content: 'Bonjour' }, ... ] )
   */
  messages = signal<ChatMessage[]>([]);

  /*
   * Indicateur d’envoi (disable input/bouton)
   */
  sending = signal(false);

  /*
   * Saisie actuelle (champ input)
   */
  currentInput = signal('');

  /*
   * showForm => True => on affiche le form user-profile
   *          => False => on affiche le résumé du profil
   */
  showForm = signal(true);

  /*
   * Reference au conteneur DOM de l’historique,
   * pour le scroll automatique.
   */
  @ViewChild('chatHistoryContainer')
  chatHistoryContainer!: ElementRef<HTMLDivElement>;

  /*
   * Endpoints sur le backend
   */
  private API_BASE_URL = 'http://localhost:8000';

  constructor(private http: HttpClient) {
    /*
      On crée un effet sur messages() :
      A chaque fois que messages() change, on appelle scrollToBottom().
      L’effet est un Reactif Angular Signals
    */
    effect(() => {
      const _ = this.messages();  // lecture du signal
      this.scrollToBottom();      // on scroll automatiquement
    });
  }

  ngOnInit(): void {
    // Pas de logique particulière ici
  }

  ngAfterViewChecked() {
    /*
      Au moment où la vue est vérifiée,
      si on a un profil défini, on scroll en bas
      (pour être sûr qu'un message ajouté est visible).
    */
    if (this.profile()) {
      this.scrollToBottom();
    }
  }

  /*
   * Toggle entre formulaire et résumé
   */
  toggleForm(): void {
    this.showForm.set(!this.showForm());
  }

  /*
   * Quand l’utilisateur soumet son profil
   *  - on stocke le nouveau UserProfile
   *  - on masque le formulaire (showForm = false)
   *  - on lance la recommandation (/recommend)
   */
  onProfileSubmit(profileData: UserProfile): void {
    this.profile.set(profileData);
    this.showForm.set(false);
    this.getRecommendation();
  }

  /*
   * getRecommendation : appelle /recommend avec le profil,
   * puis insère un message assistant dans l'historique.
   */
  private getRecommendation(): void {
    const p = this.profile();
    if (!p) return;

    this.sending.set(true);

    this.http.post<{
      recommended_course: string;
      reply: string;
      details?: any;
    }>(
      `${this.API_BASE_URL}/recommend`,
      { profile: p }
    ).subscribe({
      next: (response) => {
        this.sending.set(false);
        if (response) {
          // Met à jour le profil (recommended_course)
          const updated = {
            ...p,
            recommended_course: response.recommended_course
          } as UserProfile;
          this.profile.set(updated);

          // Construire un message assistant
          const formattedReply: ChatMessage = {
            role: 'assistant',
            content: JSON.stringify({
              reply: response.reply,
              course: response.recommended_course,
              details: response.details
            })
          };

          // Ajout à l'historique
          this.messages.update(msgs => [...msgs, formattedReply]);
        }
      },
      error: () => {
        this.sending.set(false);
        this.messages.update(msgs => [
          ...msgs,
          {
            role: 'assistant',
            content: 'Erreur lors de la recommandation.'
          }
        ]);
      }
    });
  }

  /*
   * Envoi d'un message user à /query :
   *   1) Ajoute le message user localement
   *   2) Appelle l'API
   *   3) Ajoute la réponse assistant
   */
  sendMessage(): void {
    const p = this.profile();
    if (!p) return; // si pas de profil, rien
    const inputText = this.currentInput().trim();
    if (!inputText) return; // vide ?

    // Ajout local
    this.messages.update(msgs => [
      ...msgs,
      { role: 'user', content: inputText }
    ]);

    const payload = {
      profile: p,
      history: this.messages().slice(0, -1), // On exclut le message user courant
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
          // message assistant
          this.messages.update(msgs => [
            ...msgs,
            { role: 'assistant', content: res.reply }
          ]);
        }
        // Reset champ input
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

  /*
   * scrollToBottom :
   * place l'ascenseur tout en bas pour voir le dernier message
   */
  private scrollToBottom(): void {
    try {
      // Vérifie qu'on a bien la référence
      if (!this.chatHistoryContainer?.nativeElement) return;
      const container = this.chatHistoryContainer.nativeElement;
      container.scrollTop = container.scrollHeight;
    } catch (e) {
      console.error('[Scroll error]', e);
    }
  }

  sendRecapEmail(): void {
    const p = this.profile();
    if (!p) return;
  
    // Vérifie que l'utilisateur a un email
    if (!p.email) {
      this.messages.update(msgs => [
        ...msgs,
        { role: 'assistant', content: "Veuillez renseigner un email avant l’envoi du récap." }
      ]);
      return;
    }
  
    // Appel HTTP vers /send-email
    this.http.post<{ status: string }>(
      `${this.API_BASE_URL}/send-email`, 
      p
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