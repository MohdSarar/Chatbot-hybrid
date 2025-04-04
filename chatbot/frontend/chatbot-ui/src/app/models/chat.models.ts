/*
  chat.models.ts
  Interface ChatMessage (role + content).
*/

export interface ChatMessage {
  role: 'user' | 'assistant';
  content: string;
}
