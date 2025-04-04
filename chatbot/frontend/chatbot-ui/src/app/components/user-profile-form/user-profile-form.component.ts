/*
  user-profile-form.component.ts
  Composant standalone Angular pour la saisie ou modification du profil.
  - FormGroup : name, objective, level, knowledge
  - @Input() existingProfile pour pré-remplir en mode édition
  - (profileSubmit) émis lors de la validation
*/

import { Component, Input, Output, EventEmitter, OnInit, OnChanges, SimpleChanges } from '@angular/core';
import { FormBuilder, FormGroup, Validators, ReactiveFormsModule } from '@angular/forms';
import { CommonModule } from '@angular/common';
import { UserProfile } from '../../models/user-profile.model';

@Component({
  selector: 'app-user-profile-form',
  standalone: true,
  imports: [CommonModule, ReactiveFormsModule],
  templateUrl: './user-profile-form.component.html',
  styleUrls: ['./user-profile-form.component.scss']
})
export class UserProfileFormComponent implements OnInit, OnChanges {
  /*
    existingProfile?: UserProfile | null
    Permet au parent de transmettre le profil actuel en mode édition.
  */
  @Input() existingProfile?: UserProfile | null;

  /*
    Événement émis lors de la validation.
  */
  @Output() profileSubmit = new EventEmitter<UserProfile>();

  profileForm!: FormGroup;

  levelOptions = ['Débutant', 'Intermédiaire', 'Avancé'];

  constructor(private fb: FormBuilder) {}

  ngOnInit(): void {
    this.initForm();
  }

  ngOnChanges(changes: SimpleChanges): void {
    if (changes['existingProfile']) {
      this.initForm();
    }
  }

  private initForm(): void {
    this.profileForm = this.fb.group({
      name: [this.existingProfile?.name || '', Validators.required],
      objective: [this.existingProfile?.objective || '', Validators.required],
      level: [this.existingProfile?.level || 'Débutant', Validators.required],
      knowledge: [this.existingProfile?.knowledge || '']
    });
  }

  onSubmit(): void {
    if (this.profileForm.valid) {
      const userProfile: UserProfile = {
        name: this.profileForm.value.name,
        objective: this.profileForm.value.objective,
        level: this.profileForm.value.level,
        knowledge: this.profileForm.value.knowledge,
        recommended_course: this.existingProfile?.recommended_course
      };
      this.profileSubmit.emit(userProfile);
    }
  }
}
