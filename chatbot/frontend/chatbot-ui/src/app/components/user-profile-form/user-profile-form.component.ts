import { Component, EventEmitter, Output, Input, OnInit } from '@angular/core';
import { FormBuilder, FormGroup, Validators, ReactiveFormsModule } from '@angular/forms';
import { CommonModule } from '@angular/common';
import { UserProfile } from '../../models/user-profile.model';
import { HttpClient, HttpClientModule } from '@angular/common/http';

@Component({
  selector: 'app-user-profile-form',
  standalone: true,
  imports: [CommonModule, ReactiveFormsModule, HttpClientModule],
  templateUrl: './user-profile-form.component.html',
  styleUrls: ['./user-profile-form.component.scss']
})
export class UserProfileFormComponent implements OnInit {
  @Output() profileSubmit = new EventEmitter<UserProfile>();

  /*
    existingProfile?: UserProfile | null
    Permet au parent de transmettre le profil actuel en mode édition.
  */
  @Input() existingProfile?: UserProfile | null;

  profileForm!: FormGroup;
  levelOptions = ['Débutant', 'Intermédiaire', 'Avancé'];
  selectedFileName: string | null = null;
  uploadedContent: string = ''; // Contenu texte extrait du PDF

  constructor(private fb: FormBuilder, private http: HttpClient) {}

  ngOnInit(): void {
    this.profileForm = this.fb.group({
      name: ['', Validators.required],
      objective: ['', Validators.required],
      level: ['Débutant', Validators.required],
      knowledge: ['']  // Champ texte libre
    });
  }

  onFileSelected(event: Event): void {
    const input = event.target as HTMLInputElement;
    if (!input.files?.length) return;

    const file = input.files[0];
    this.selectedFileName = file.name;

    const formData = new FormData();
    formData.append('file', file);

    this.http.post<{ content: string }>(
      'http://localhost:8000/upload-pdf',
      formData
    ).subscribe({
      next: (res) => {
        this.uploadedContent = res.content;
        console.log('[PDF upload] Contenu extrait :', this.uploadedContent);
      },
      error: (err) => {
        console.error('[PDF upload] Erreur :', err);
      }
    });
  }

  onSubmit(): void {
    if (this.profileForm.valid) {
      const userProfile: UserProfile = {
        name: this.profileForm.value.name,
        objective: this.profileForm.value.objective,
        level: this.profileForm.value.level,
        knowledge: this.profileForm.value.knowledge,
        pdf_content: this.uploadedContent // ✅ On envoie le texte extrait
      };
      this.profileSubmit.emit(userProfile);
    }
  }
  
}
