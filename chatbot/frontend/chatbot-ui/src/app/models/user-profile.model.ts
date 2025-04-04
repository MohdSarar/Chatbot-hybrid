/*
  user-profile.model.ts
  Décrit le modèle du profil utilisateur.
*/

export interface UserProfile {
  name: string;
  objective: string;
  level: string;
  knowledge: string;  // Champ libre
  recommended_course?: string;
}
