# ✍️ Handwritten Page Generator (Conditional GAN)

A web application that synthesizes realistic **handwritten notebook pages, notes, and paragraphs** directly from typed text using a pre-trained **Conditional Generative Adversarial Network (cGAN)** trained on the **EMNIST Balanced** dataset.

---

## 🌟 Key Features

- **Side-by-Side Dual-Pane Interface**:
  - **Left Side**: Full-page text editor with instant controls.
  - **Right Side**: Real-time rendering of a realistic **Handwritten Page**.
- **Full Page & Paragraph Support**: Automatic multi-line word wrapping and paragraph preservation.
- **Adjustable Text Size (Font Size)**: Dedicated slider to scale character font size from small to large.
- **Notebook Paper Styles**:
  - 📜 **Ruled Notebook Paper**: Classic ruled blue lines with a red left vertical margin.
  - 📄 **Plain White Paper**: Clean, borderless white sheet.
  - 🌑 **Dark Canvas**: High-contrast white ink on dark background.
- **Ink Color Options**: Toggle between **Classic Blue ink** and **Dark Black ink**.
- **Consistent vs. Natural Penmanship**: Choose between uniform handwriting style across the page or organic variations per letter.
- **One-Click Download**: Export high-resolution PNG of the entire synthesized handwritten page.

---

## 🚀 Running Locally

```powershell
# 1. Activate virtual environment
.\venv\Scripts\activate

# 2. Install dependencies (PyTorch CPU wheel keeps install light)
pip install -r requirements.txt

# 3. Launch the Streamlit app
streamlit run app.py
```
Open **`http://localhost:8501`** in your browser.

---

## 🧠 Model Architecture

- **Model**: Conditional DCGAN (`cDCGAN`) trained on EMNIST Balanced.
- **Conditioning**: Learned class embeddings ($y \in \{0, \dots, 46\}$) combined with Gaussian latent noise $z \in \mathcal{N}(0, I_{100})$.
- **Classes**: 47 classes (Digits `0-9`, Uppercase `A-Z`, and balanced lowercase `abdefghnqrt`). Non-supported lowercase letters are automatically mapped to uppercase.
- **Checkpoint**: Loaded directly from `generator.pt` on CPU.

---

## 📦 Exact Git Commands to Push to GitHub

```powershell
# 1. Initialize git
git init

# 2. Stage tracked files
git add .gitignore app.py generator.pt requirements.txt README.md

# 3. Commit
git commit -m "feat: side-by-side full page handwritten text generator"

# 4. Set main branch
git branch -M main

# 5. Link GitHub repo (replace with your repo URL)
git remote add origin https://github.com/<your-username>/<your-repo-name>.git

# 6. Push to GitHub
git push -u origin main
```
