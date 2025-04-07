// user-profile.model.ts

export interface UserProfile {
  name: string;
  objective: string;
  level: string;
  knowledge: string;

  email?: string;         // <-- nouveau champ optionnel
  pdf_content?: string;
  recommended_course?: string;
}
