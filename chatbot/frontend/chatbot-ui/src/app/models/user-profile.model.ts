/*
  user-profile.model.ts
  Décrit le modèle du profil utilisateur.
*/

export interface UserProfile {
  name: string;
  objective: string;
  level: string;
  knowledge: string;
  pdf_content?: string; // Nouveau champ pour inclure le texte extrait du fichier PDF
  recommended_course?: string;
}
