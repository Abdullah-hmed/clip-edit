import tkinter as tk
from tkinter import ttk, scrolledtext, filedialog
from PIL import Image, ImageTk
import torch
import numpy as np
from threading import Thread
import os
import sys
import glob

# Add StyleGAN path
current_dir = os.getcwd()
sys.path.append(os.path.join(current_dir, "stylegan2-ada-pytorch"))

from legacy import load_network_pkl
from torchvision import transforms
import clip
from tqdm import tqdm

class StyleGANEditor:
    def __init__(self, root):
        self.root = root
        self.root.title("StyleGAN2 CLIP Editor")
        self.root.geometry("1200x800")
        
        # Model loading status
        self.models_loaded = False
        self.current_w = None
        self.w_history = []  # History stack of W values
        self.current_img = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Setup UI
        self.setup_ui()
        
        # Find available .pkl files
        self.find_pkl_files()
    
    def find_pkl_files(self):
        """Find all .pkl files in current directory and subdirectories"""
        pkl_files = []
        
        # Search in current directory
        pkl_files.extend(glob.glob("*.pkl"))
        
        # Search in common subdirectories
        common_dirs = ["stylegan2_ada_pytorch", "pretrained-weights", "models", "checkpoints"]
        for dir_name in common_dirs:
            if os.path.exists(dir_name):
                pkl_files.extend(glob.glob(os.path.join(dir_name, "*.pkl")))
                pkl_files.extend(glob.glob(os.path.join(dir_name, "**", "*.pkl"), recursive=True))
        
        # Remove duplicates and sort
        pkl_files = sorted(list(set(pkl_files)))
        
        # Update dropdown
        if pkl_files:
            self.model_dropdown['values'] = pkl_files
            self.model_dropdown.current(0)
        else:
            self.model_dropdown['values'] = ["No .pkl files found"]
    
    def browse_pkl_file(self):
        """Open file browser to select a .pkl file"""
        filename = filedialog.askopenfilename(
            title="Select StyleGAN2 model file",
            filetypes=[("Pickle files", "*.pkl"), ("All files", "*.*")]
        )
        if filename:
            current_values = list(self.model_dropdown['values'])
            if filename not in current_values:
                current_values.append(filename)
                self.model_dropdown['values'] = current_values
            self.model_dropdown.set(filename)
    
    def setup_ui(self):
        # Main container
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(1, weight=1)
        
        # Left panel - Controls
        control_frame = ttk.LabelFrame(main_frame, text="Controls", padding="10")
        control_frame.grid(row=0, column=0, rowspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 10))
        
        # Model selection section
        model_frame = ttk.Frame(control_frame)
        model_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        ttk.Label(model_frame, text="Model:").grid(row=0, column=0, sticky=tk.W, pady=(0, 5))
        
        model_select_frame = ttk.Frame(model_frame)
        model_select_frame.grid(row=1, column=0, sticky=(tk.W, tk.E))
        model_select_frame.columnconfigure(0, weight=1)
        
        self.model_dropdown = ttk.Combobox(model_select_frame, state="readonly", width=25)
        self.model_dropdown.grid(row=0, column=0, sticky=(tk.W, tk.E), padx=(0, 5))
        
        browse_btn = ttk.Button(model_select_frame, text="Browse...", command=self.browse_pkl_file, width=10)
        browse_btn.grid(row=0, column=1)
        
        # Load model button
        self.load_model_btn = ttk.Button(control_frame, text="Load Model", 
                                         command=self.load_models_thread)
        self.load_model_btn.grid(row=1, column=0, columnspan=2, pady=5, sticky=(tk.W, tk.E))
        
        # Status label
        self.status_label = ttk.Label(control_frame, text="Select a model and click Load Model", foreground="blue")
        self.status_label.grid(row=2, column=0, columnspan=2, pady=(0, 10))
        
        # Separator
        ttk.Separator(control_frame, orient=tk.HORIZONTAL).grid(row=3, column=0, columnspan=2, 
                                                                 sticky=(tk.W, tk.E), pady=10)
        
        # Generate button
        self.generate_btn = ttk.Button(control_frame, text="Generate New Image", 
                                       command=self.generate_image, state=tk.DISABLED)
        self.generate_btn.grid(row=4, column=0, columnspan=2, pady=5, sticky=(tk.W, tk.E))
        
        # Undo button
        self.undo_btn = ttk.Button(control_frame, text="Undo", 
                                   command=self.undo_edit, state=tk.DISABLED)
        self.undo_btn.grid(row=5, column=0, columnspan=2, pady=5, sticky=(tk.W, tk.E))
        
        # Save comparison button
        self.save_btn = ttk.Button(control_frame, text="Save Comparison", 
                                   command=self.save_comparison, state=tk.DISABLED)
        self.save_btn.grid(row=6, column=0, columnspan=2, pady=5, sticky=(tk.W, tk.E))
        
        # History label
        self.history_label = ttk.Label(control_frame, text="History: 0 states")
        self.history_label.grid(row=7, column=0, columnspan=2, pady=(5, 0))
        
        # Truncation slider
        ttk.Label(control_frame, text="Truncation:").grid(row=8, column=0, sticky=tk.W, pady=(10, 0))
        self.truncation_var = tk.DoubleVar(value=0.7)
        self.truncation_slider = ttk.Scale(control_frame, from_=0.0, to=1.0, 
                                           variable=self.truncation_var, orient=tk.HORIZONTAL)
        self.truncation_slider.grid(row=9, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 5))
        self.truncation_label = ttk.Label(control_frame, text="0.70")
        self.truncation_label.grid(row=10, column=0, columnspan=2)
        self.truncation_var.trace_add("write", self.update_truncation_label)
        
        # Separator
        ttk.Separator(control_frame, orient=tk.HORIZONTAL).grid(row=11, column=0, columnspan=2, 
                                                                 sticky=(tk.W, tk.E), pady=15)
        
        # CLIP Edit section
        ttk.Label(control_frame, text="CLIP Editing", font=('TkDefaultFont', 10, 'bold')).grid(
            row=12, column=0, columnspan=2, sticky=tk.W, pady=(0, 10))
        
        # Source prompt
        ttk.Label(control_frame, text="Source prompt:").grid(row=13, column=0, sticky=tk.W)
        self.source_entry = ttk.Entry(control_frame, width=30)
        self.source_entry.grid(row=14, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 5))
        self.source_entry.insert(0, "a person")
        
        # Target prompt
        ttk.Label(control_frame, text="Target prompt:").grid(row=15, column=0, sticky=tk.W)
        self.target_entry = ttk.Entry(control_frame, width=30)
        self.target_entry.grid(row=16, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 5))
        self.target_entry.insert(0, "a smiling person")
        
        # Unwanted prompt
        ttk.Label(control_frame, text="Unwanted (optional):").grid(row=17, column=0, sticky=tk.W)
        self.unwanted_entry = ttk.Entry(control_frame, width=30)
        self.unwanted_entry.grid(row=18, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # Steps slider
        ttk.Label(control_frame, text="Edit steps:").grid(row=19, column=0, sticky=tk.W)
        self.steps_var = tk.IntVar(value=20)
        self.steps_slider = ttk.Scale(control_frame, from_=5, to=50, 
                                      variable=self.steps_var, orient=tk.HORIZONTAL)
        self.steps_slider.grid(row=20, column=0, columnspan=2, sticky=(tk.W, tk.E))
        self.steps_label = ttk.Label(control_frame, text="20")
        self.steps_label.grid(row=21, column=0, columnspan=2, pady=(0, 5))
        self.steps_var.trace_add("write", self.update_steps_label)
        
        # L2 coefficient
        ttk.Label(control_frame, text="L2 strength:").grid(row=22, column=0, sticky=tk.W)
        self.l2_var = tk.DoubleVar(value=100.0)
        self.l2_slider = ttk.Scale(control_frame, from_=10.0, to=500.0, 
                                   variable=self.l2_var, orient=tk.HORIZONTAL)
        self.l2_slider.grid(row=23, column=0, columnspan=2, sticky=(tk.W, tk.E))
        self.l2_label = ttk.Label(control_frame, text="100.0")
        self.l2_label.grid(row=24, column=0, columnspan=2, pady=(0, 10))
        self.l2_var.trace_add("write", self.update_l2_label)
        
        # Edit button
        self.edit_btn = ttk.Button(control_frame, text="Apply CLIP Edit", 
                                   command=self.apply_edit, state=tk.DISABLED)
        self.edit_btn.grid(row=25, column=0, columnspan=2, pady=5, sticky=(tk.W, tk.E))
        
        # Progress label
        self.progress_label = ttk.Label(control_frame, text="")
        self.progress_label.grid(row=26, column=0, columnspan=2, pady=(5, 0))
        
        # Right panel - Image display
        image_frame = ttk.LabelFrame(main_frame, text="Generated Image", padding="10")
        image_frame.grid(row=0, column=1, rowspan=2, sticky=(tk.W, tk.E, tk.N, tk.S))
        image_frame.columnconfigure(0, weight=1)
        image_frame.rowconfigure(0, weight=1)
        
        # Canvas for image
        self.canvas = tk.Canvas(image_frame, bg='gray', width=512, height=512)
        self.canvas.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Placeholder text
        self.canvas.create_text(256, 256, text="No image generated yet", 
                               fill="white", font=('TkDefaultFont', 12))
    
    def update_truncation_label(self, *args):
        self.truncation_label.config(text=f"{self.truncation_var.get():.2f}")
    
    def update_steps_label(self, *args):
        self.steps_label.config(text=f"{self.steps_var.get()}")
    
    def update_l2_label(self, *args):
        self.l2_label.config(text=f"{self.l2_var.get():.1f}")
    
    def update_history_label(self):
        """Update the history label to show current state"""
        num_states = len(self.w_history)
        self.history_label.config(text=f"History: {num_states} state{'s' if num_states != 1 else ''}")
    
    def load_models_thread(self):
        thread = Thread(target=self.load_models, daemon=True)
        thread.start()
    
    def load_models(self):
        try:
            selected_model = self.model_dropdown.get()
            
            if not selected_model or selected_model == "No .pkl files found":
                self.status_label.config(text="Please select a valid .pkl file", foreground="red")
                return
            
            if not os.path.exists(selected_model):
                self.status_label.config(text=f"File not found: {selected_model}", foreground="red")
                return
            
            # Disable controls during loading
            self.load_model_btn.config(state=tk.DISABLED)
            self.model_dropdown.config(state=tk.DISABLED)
            
            self.status_label.config(text="Loading CLIP model...", foreground="orange")
            self.clip_model, self.preprocess = clip.load("ViT-B/32", device=self.device)
            
            self.status_label.config(text=f"Loading StyleGAN2 model from {selected_model}...", foreground="orange")
            with open(selected_model, "rb") as f:
                self.g_ema = load_network_pkl(f)["G_ema"].to(self.device)
            
            self.models_loaded = True
            self.status_label.config(text=f"Models loaded successfully!", foreground="green")
            self.generate_btn.config(state=tk.NORMAL)
            
            # Re-enable controls
            self.load_model_btn.config(state=tk.NORMAL)
            self.model_dropdown.config(state="readonly")
            
        except Exception as e:
            self.status_label.config(text=f"Error loading models: {str(e)}", foreground="red")
            self.load_model_btn.config(state=tk.NORMAL)
            self.model_dropdown.config(state="readonly")
    
    def generate_image(self):
        if not self.models_loaded:
            return
        
        thread = Thread(target=self._generate_image, daemon=True)
        thread.start()
    
    def _generate_image(self):
        try:
            self.generate_btn.config(state=tk.DISABLED)
            self.edit_btn.config(state=tk.DISABLED)
            self.undo_btn.config(state=tk.DISABLED)
            self.save_btn.config(state=tk.DISABLED)
            self.progress_label.config(text="Generating...")
            
            # Generate random latent
            z = torch.randn([1, self.g_ema.z_dim], device=self.device)
            c = torch.zeros([1, self.g_ema.c_dim], device=self.device)
            w = self.g_ema.mapping(z, c)
            
            # Apply truncation
            truncation = self.truncation_var.get()
            w_avg = self.g_ema.mapping.w_avg
            w = w_avg + truncation * (w - w_avg)
            
            self.current_w = w
            
            # Reset history with new generation
            self.w_history = [w.clone()]
            self.update_history_label()
            
            # Generate image
            with torch.no_grad():
                img = self.g_ema.synthesis(w)
                img = (img.clamp(-1, 1) + 1) / 2
            
            self.current_img = img
            self.display_image(img)
            
            self.progress_label.config(text="Image generated!")
            self.edit_btn.config(state=tk.NORMAL)
            self.save_btn.config(state=tk.NORMAL)
            # Undo button stays disabled until an edit is applied
            
        except Exception as e:
            self.progress_label.config(text=f"Error: {str(e)}")
        finally:
            self.generate_btn.config(state=tk.NORMAL)
    
    def undo_edit(self):
        if not self.models_loaded or len(self.w_history) <= 1:
            return
        
        thread = Thread(target=self._undo_edit, daemon=True)
        thread.start()
    
    def _undo_edit(self):
        try:
            self.generate_btn.config(state=tk.DISABLED)
            self.edit_btn.config(state=tk.DISABLED)
            self.undo_btn.config(state=tk.DISABLED)
            self.save_btn.config(state=tk.DISABLED)
            self.progress_label.config(text="Undoing...")
            
            # Remove current state from history
            self.w_history.pop()
            
            # Get previous state
            self.current_w = self.w_history[-1].clone()
            self.update_history_label()
            
            # Generate image from previous state
            with torch.no_grad():
                img = self.g_ema.synthesis(self.current_w)
                img = (img.clamp(-1, 1) + 1) / 2
            
            self.current_img = img
            self.display_image(img)
            
            self.progress_label.config(text="Undone!")
            
        except Exception as e:
            self.progress_label.config(text=f"Error: {str(e)}")
        finally:
            self.generate_btn.config(state=tk.NORMAL)
            self.edit_btn.config(state=tk.NORMAL)
            self.save_btn.config(state=tk.NORMAL)
            # Enable undo only if there's more history
            if len(self.w_history) > 1:
                self.undo_btn.config(state=tk.NORMAL)
    
    def save_comparison(self):
        if not self.models_loaded or len(self.w_history) == 0:
            return
        
        thread = Thread(target=self._save_comparison, daemon=True)
        thread.start()
    
    def _save_comparison(self):
        try:
            self.progress_label.config(text="Saving comparison...")
            
            import matplotlib
            matplotlib.use('Agg')  # Use non-interactive backend
            import matplotlib.pyplot as plt
            from datetime import datetime
            
            # Generate original image (first in history)
            with torch.no_grad():
                original_img = self.g_ema.synthesis(self.w_history[0])
                original_img = (original_img.clamp(-1, 1) + 1) / 2
                original_np = original_img[0].permute(1, 2, 0).cpu().numpy()
            
            # Get current image
            current_np = self.current_img[0].permute(1, 2, 0).cpu().numpy()
            
            # Create figure
            fig, axes = plt.subplots(1, 2, figsize=(12, 6))
            
            axes[0].imshow(original_np)
            axes[0].set_title("Original", fontsize=14, fontweight='bold')
            axes[0].axis('off')
            
            axes[1].imshow(current_np)
            axes[1].set_title("Edited", fontsize=14, fontweight='bold')
            axes[1].axis('off')
            
            plt.tight_layout()
            
            # Save with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"comparison_{timestamp}.png"
            plt.savefig(filename, dpi=150, bbox_inches='tight')
            plt.close()
            
            self.progress_label.config(text=f"Saved as {filename}!")
            
        except Exception as e:
            self.progress_label.config(text=f"Error saving: {str(e)}")
    
    def display_image(self, img_tensor):
        # Convert tensor to PIL Image
        img_np = img_tensor[0].permute(1, 2, 0).cpu().numpy()
        img_np = (img_np * 255).astype(np.uint8)
        pil_img = Image.fromarray(img_np)
        
        # Resize to fit canvas
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        if canvas_width > 1 and canvas_height > 1:
            pil_img = pil_img.resize((min(512, canvas_width), min(512, canvas_height)), 
                                     Image.Resampling.LANCZOS)
        
        # Convert to PhotoImage
        photo = ImageTk.PhotoImage(pil_img)
        
        # Display on canvas
        self.canvas.delete("all")
        self.canvas.create_image(canvas_width//2 if canvas_width > 1 else 256, 
                                canvas_height//2 if canvas_height > 1 else 256, 
                                image=photo)
        self.canvas.image = photo  # Keep reference
    
    def apply_edit(self):
        if not self.models_loaded or self.current_w is None:
            return
        
        thread = Thread(target=self._apply_edit, daemon=True)
        thread.start()
    
    def _apply_edit(self):
        try:
            self.generate_btn.config(state=tk.DISABLED)
            self.edit_btn.config(state=tk.DISABLED)
            self.undo_btn.config(state=tk.DISABLED)
            self.save_btn.config(state=tk.DISABLED)
            
            p_source = self.source_entry.get()
            p_target = self.target_entry.get()
            p_unwanted = self.unwanted_entry.get()
            steps = self.steps_var.get()
            l2_coeff = self.l2_var.get()
            
            # Encode text prompts
            with torch.no_grad():
                t_source = self.clip_model.encode_text(clip.tokenize([p_source]).to(self.device))
                t_target = self.clip_model.encode_text(clip.tokenize([p_target]).to(self.device))
                t_source = t_source / t_source.norm(dim=-1, keepdim=True)
                t_target = t_target / t_target.norm(dim=-1, keepdim=True)
                
                edit_dir = (t_target - t_source).detach()
                
                if p_unwanted.strip():
                    t_unwanted = self.clip_model.encode_text(clip.tokenize([p_unwanted]).to(self.device))
                    t_unwanted = t_unwanted / t_unwanted.norm(dim=-1, keepdim=True)
                    edit_dir = edit_dir - t_unwanted
                
                edit_dir = edit_dir / edit_dir.norm(dim=-1, keepdim=True)
            
            # Initialize W+ from current W (top of history stack)
            w_plus = self.current_w.clone().detach().requires_grad_(True)
            optimizer = torch.optim.AdamW([w_plus], lr=0.01, betas=(0.9, 0.999))
            
            preprocess_clip = transforms.Normalize(
                mean=(0.48145466, 0.4578275, 0.40821073),
                std=(0.26862954, 0.26130258, 0.27577711)
            )
            
            # Optimization loop
            for step in range(steps):
                self.progress_label.config(text=f"Editing: {step+1}/{steps}")
                
                optimizer.zero_grad()
                img = self.g_ema.synthesis(w_plus)
                img_norm = (img.clamp(-1, 1) + 1) / 2
                
                # Random crops for CLIP
                crops = self.random_crops(img_norm, out_size=224, big_size=256, n=8)
                crops = preprocess_clip(crops).to(self.device)
                embed_img = self.clip_model.encode_image(crops)
                embed_img = embed_img / embed_img.norm(dim=-1, keepdim=True)
                embed_mean = embed_img.mean(dim=0, keepdim=True)
                
                # Directional CLIP loss
                delta_img = embed_mean - t_source
                delta_text = edit_dir
                loss_dir = 1 - torch.cosine_similarity(delta_img, delta_text, dim=-1).mean()
                
                # Regularization to original (first in history)
                loss_reg = l2_coeff * (w_plus - self.w_history[0].detach()).pow(2).mean()
                
                loss = loss_dir + loss_reg
                loss.backward()
                optimizer.step()
            
            # Generate final image
            with torch.no_grad():
                final_img = self.g_ema.synthesis(w_plus)
                final_img = (final_img.clamp(-1, 1) + 1) / 2
            
            # Update current state and add to history
            self.current_w = w_plus.detach()
            self.w_history.append(self.current_w.clone())
            self.update_history_label()
            
            self.current_img = final_img
            self.display_image(final_img)
            
            self.progress_label.config(text="Edit complete!")
            self.undo_btn.config(state=tk.NORMAL)  # Enable undo after edit
            
        except Exception as e:
            self.progress_label.config(text=f"Error: {str(e)}")
        finally:
            self.generate_btn.config(state=tk.NORMAL)
            self.edit_btn.config(state=tk.NORMAL)
            self.save_btn.config(state=tk.NORMAL)
    
    def random_crops(self, img_norm, out_size=224, big_size=256, n=6):
        img_big = torch.nn.functional.interpolate(
            img_norm, size=(big_size, big_size),
            mode='bilinear', align_corners=False
        )
        B, C, H, W = img_big.shape
        crops = []
        for _ in range(n):
            top = torch.randint(0, H - out_size + 1, (1,)).item()
            left = torch.randint(0, W - out_size + 1, (1,)).item()
            crop = img_big[:, :, top:top+out_size, left:left+out_size]
            crops.append(crop)
        return torch.cat(crops, dim=0)

if __name__ == "__main__":
    root = tk.Tk()
    app = StyleGANEditor(root)
    root.mainloop()