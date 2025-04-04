/**
 * main.ts
 * Point d'entrée principal de l'application Angular standalone.
 * On bootstrap directement ChatComponent.
 */
import { bootstrapApplication } from '@angular/platform-browser';
import { ChatComponent } from './app/components/chat/chat.component';

// Optionnel: app.config.ts si tu veux configurer un router ou autre
// import { appConfig } from './app.config';

bootstrapApplication(ChatComponent /* , appConfig */)
  .catch(err => console.error(err));
