/**
 * user-profile-form.component.ts
 * 
 * Composant Angular standalone représentant le formulaire de profil utilisateur.
 * - Permet de saisir nom, objectif, niveau, compétences, email.
 * - Inclut le téléversement d’un fichier PDF (optionnel).
 * - Émet un événement `profileSubmit` avec le profil complet (incluant le contenu PDF).
 */

import { Component, EventEmitter, Output, Input, OnInit } from '@angular/core';
import { FormBuilder, FormGroup, Validators, ReactiveFormsModule } from '@angular/forms';
import { CommonModule } from '@angular/common';
import { HttpClient, HttpClientModule } from '@angular/common/http';
import { finalize } from 'rxjs/operators';

// L'interface contenant name, email, objective, level, ...
import { UserProfile } from '../../models/user-profile.model';

@Component({
  selector: 'app-user-profile-form',
  standalone: true,
  imports: [CommonModule, ReactiveFormsModule, HttpClientModule],
  templateUrl: './user-profile-form.component.html',
  styleUrls: ['./user-profile-form.component.scss']
})
export class UserProfileFormComponent implements OnInit {
  /** Événement émis lors de la soumission du formulaire */
  @Output() profileSubmit = new EventEmitter<UserProfile>();

  /** Profil existant (édition) */
  @Input() existingProfile?: UserProfile | null;

  /** Formulaire réactif */
  profileForm!: FormGroup;

  /** Liste des niveaux disponibles */
  levelOptions = ['Débutant', 'Intermédiaire', 'Avancé'];

  /** Indique si un chargement PDF est en cours */
  loadingPdf: boolean = false;

  /** Nom du fichier PDF sélectionné (facultatif) */
  selectedFileName: string | null = null;

  /** Contenu texte extrait du PDF */
  uploadedContent: string = '';

  constructor(
    private fb: FormBuilder,
    private http: HttpClient
  ) {}

  /** Initialisation du formulaire */
  ngOnInit(): void {
    /*
     * On définit les champs du formulaire :
     *  - name, objective, level, knowledge requis ou non
     *  - email facultatif
     */
    this.profileForm = this.fb.group({
      name: ['', Validators.required],
      email: [''],                       // email facultatif (ex: Validators.email si besoin)
      objective: ['', Validators.required],
      level: ['Débutant', Validators.required],
      knowledge: ['']
    });

    // Si un profil existant est passé en @Input, on pré-remplit
    if (this.existingProfile) {
      this.profileForm.patchValue(this.existingProfile);
    }
  }

  /**
   * Gère la sélection d’un fichier PDF dans l’input
   * -> Envoi à /upload-pdf
   * -> Stocke la réponse dans uploadedContent
   */
  onFileSelected(event: Event): void {
    const input = event.target as HTMLInputElement;
    if (!input.files?.length) return;

    const file = input.files[0];
    this.selectedFileName = file.name;

    const formData = new FormData();
    formData.append('file', file);

    // Active le loader
    this.loadingPdf = true;

    // Simule un délai de 1.2 secondes pour effet UX
    setTimeout(() => {
      this.http.post<{ content: string }>(
        'http://localhost:8000/upload-pdf',
        formData
      )
      .pipe(finalize(() => this.loadingPdf = false))
      .subscribe({
        next: (res) => {
          this.uploadedContent = res.content;
          console.log('[PDF upload] Contenu extrait :', this.uploadedContent);
        },
        error: (err) => {
          console.error('[PDF upload] Erreur :', err);
        }
      });
    }, 1200);
  }

  /**
   * Soumission du formulaire :
   * - Construit un UserProfile
   * - Emet l'événement profileSubmit
   */
  onSubmit(): void {
    if (this.profileForm.valid) {
      const userProfile: UserProfile = {
        name: this.profileForm.value.name,
        email: this.profileForm.value.email,        // on stocke l'email du formulaire
        objective: this.profileForm.value.objective,
        level: this.profileForm.value.level,
        knowledge: this.profileForm.value.knowledge,
        pdf_content: this.uploadedContent
      };
      // On émet l'événement avec ce profil complet
      this.profileSubmit.emit(userProfile);
    }
  }
}
