import io
import os
import numpy as np
from PIL import Image, ImageDraw
import streamlit as st
import torch
import torch.nn as nn

# -----------------------------------------------------------------------------
# 1. MODEL ARCHITECTURE (Must match generator.pt weights)
# -----------------------------------------------------------------------------
LATENT_DIM = 100
NUM_CLASSES = 47
CLASS_CHARS = list("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabdefghnqrt")
CHAR_TO_IDX = {ch: idx for idx, ch in enumerate(CLASS_CHARS)}
VALID_SET = set(CLASS_CHARS)


class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(NUM_CLASSES, LATENT_DIM)
        self.fc = nn.Sequential(
            nn.Linear(LATENT_DIM, 256 * 7 * 7),
            nn.BatchNorm1d(256 * 7 * 7),
            nn.ReLU(True),
        )
        self.net = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, 1, 3, 1, 1),
            nn.Tanh(),
        )

    def forward(self, z, y):
        return self.net(self.fc(z * self.emb(y)).view(-1, 256, 7, 7))


@st.cache_resource
def load_model():
    """Load pre-trained PyTorch generator weights."""
    if not os.path.exists("generator.pt"):
        return None
    g = Generator()
    g.load_state_dict(torch.load("generator.pt", map_location="cpu"))
    g.eval()
    return g


# -----------------------------------------------------------------------------
# 2. TEXT PROCESSING & MULTI-LINE WORD WRAPPING
# -----------------------------------------------------------------------------
def map_char(c: str):
    """
    Maps an input character to the nearest valid EMNIST Balanced class:
    - Direct match for digits, uppercase letters, and balanced lowercase letters.
    - Other lowercase letters are mapped to uppercase.
    - Non-supported symbols return None.
    """
    if c in VALID_SET:
        return c, None
    if c.islower():
        upper = c.upper()
        if upper in VALID_SET:
            return upper, None
    return None, c


def wrap_text_into_lines(text: str, max_chars_per_line: int = 36):
    """Wraps text into multi-line paragraphs respecting user newlines and max width."""
    lines = []
    skipped_chars = set()

    for paragraph in text.split("\n"):
        p_clean = paragraph.strip()
        if not p_clean:
            lines.append("")
            continue

        words = paragraph.split(" ")
        current_line = []
        current_len = 0

        for w in words:
            # Check characters in word
            clean_word = []
            for ch in w:
                mapped, skipped = map_char(ch)
                if mapped:
                    clean_word.append(ch)  # keep original char for spacing
                elif skipped:
                    skipped_chars.add(skipped)

            w_len = len(clean_word)
            if w_len == 0:
                continue

            space_len = 1 if current_line else 0
            if current_len + w_len + space_len <= max_chars_per_line:
                current_line.append(w)
                current_len += w_len + space_len
            else:
                if current_line:
                    lines.append(" ".join(current_line))
                current_line = [w]
                current_len = w_len

        if current_line:
            lines.append(" ".join(current_line))

    return lines, list(skipped_chars)


# -----------------------------------------------------------------------------
# 3. PAGE SYNTHESIZER
# -----------------------------------------------------------------------------
def render_handwritten_page(
    lines: list,
    model,
    seed: int = 42,
    char_size: int = 32,
    max_chars_per_line: int = 36,
    consistent: bool = True,
    paper_style: str = "Ruled Notebook",
    ink_color: str = "Classic Blue",
):
    """Synthesizes a realistic multi-line handwritten page canvas."""
    # 1. Collect all valid character tokens and calculate coordinate offsets
    items_to_render = []
    char_spacing = int(char_size * 0.82)
    space_width = int(char_size * 0.52)

    for line_idx, line in enumerate(lines):
        x_cursor = 0
        for ch in line:
            if ch == " ":
                x_cursor += space_width
            else:
                mapped, _ = map_char(ch)
                if mapped:
                    items_to_render.append({
                        "idx": CHAR_TO_IDX[mapped],
                        "line": line_idx,
                        "x": x_cursor,
                    })
                    x_cursor += char_spacing

    total_chars = len(items_to_render)
    char_imgs = []

    # 2. Batch Inference
    if total_chars > 0:
        torch.manual_seed(seed)
        if consistent:
            # Same handwriting style for all letters
            shared_z = torch.randn(1, LATENT_DIM)
            z_batch = shared_z.repeat(total_chars, 1)
        else:
            # Organic variety in handwriting
            z_batch = torch.randn(total_chars, LATENT_DIM)

        y_batch = torch.tensor([item["idx"] for item in items_to_render], dtype=torch.long)

        with torch.no_grad():
            out_tensors = (model(z_batch, y_batch) + 1.0) / 2.0  # (N, 1, 28, 28) in [0, 1]

        # Resize each character to target text size with high-quality bicubic filtering
        for i in range(total_chars):
            norm = out_tensors[i, 0].clamp(0.0, 1.0).numpy()
            raw_pil = Image.fromarray((norm * 255.0).astype(np.uint8), mode="L")
            resized = raw_pil.resize((char_size, char_size), Image.Resampling.BICUBIC)
            char_imgs.append(resized)

    # 3. Canvas Dimensions
    left_margin = 65 if paper_style == "Ruled Notebook" else 35
    right_margin = 35
    top_margin = 45
    bottom_margin = 45
    line_height = int(char_size * 1.55)

    num_lines = max(len(lines), 7)  # at least 7 lines for realistic page proportions
    page_width = left_margin + int(max_chars_per_line * char_spacing) + right_margin
    page_height = top_margin + (num_lines * line_height) + bottom_margin

    # 4. Paper Style & Background Setup
    if paper_style == "Dark Canvas":
        bg_color = (24, 28, 36)
        line_color = (45, 55, 72)
        margin_line_color = (65, 80, 105)
        ink_rgb = (245, 245, 250)
    elif paper_style == "Plain White":
        bg_color = (255, 255, 255)
        line_color = None
        margin_line_color = None
        ink_rgb = (18, 24, 38) if ink_color == "Dark Black" else (22, 54, 142)
    else:  # Ruled Notebook
        bg_color = (253, 252, 247)  # subtle ivory paper
        line_color = (212, 226, 246)  # faint blue horizontal ruling
        margin_line_color = (245, 145, 145)  # faint red left vertical margin line
        ink_rgb = (20, 26, 40) if ink_color == "Dark Black" else (24, 58, 150)

    page_img = Image.new("RGB", (page_width, page_height), color=bg_color)
    draw = ImageDraw.Draw(page_img)

    # 5. Draw Paper Ruling & Margins
    if paper_style == "Ruled Notebook":
        # Red vertical margin line
        draw.line([(left_margin - 16, 0), (left_margin - 16, page_height)], fill=margin_line_color, width=2)
        # Blue horizontal ruling lines
        for i in range(num_lines + 1):
            y_rule = top_margin + (i * line_height) + int(char_size * 0.92)
            draw.line([(0, y_rule), (page_width, y_rule)], fill=line_color, width=1)
    elif paper_style == "Dark Canvas":
        for i in range(num_lines + 1):
            y_rule = top_margin + (i * line_height) + int(char_size * 0.92)
            draw.line([(0, y_rule), (page_width, y_rule)], fill=line_color, width=1)

    # 6. Paste Characters using Alpha Masks for clean ink edges
    for item_idx, item in enumerate(items_to_render):
        c_img = char_imgs[item_idx]
        pos_x = left_margin + item["x"]
        pos_y = top_margin + (item["line"] * line_height)

        c_np = np.array(c_img, dtype=np.float32) / 255.0
        # Gentle curve adjustment to remove faint raster noise while preserving smooth ink strokes
        mask_np = np.clip((c_np - 0.12) / 0.88, 0.0, 1.0)
        alpha_channel = Image.fromarray((mask_np * 255.0).astype(np.uint8), mode="L")

        stroke_color = Image.new("RGBA", (char_size, char_size), color=(*ink_rgb, 255))
        page_img.paste(stroke_color, (pos_x, pos_y), mask=alpha_channel)

    return page_img


# -----------------------------------------------------------------------------
# 4. STREAMLIT INTERFACE (WIDE SIDE-BY-SIDE LAYOUT)
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Handwritten Page Generator",
    page_icon="✍️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
        .block-container {
            padding-top: 1.5rem;
            padding-bottom: 2rem;
            max-width: 1400px;
        }
        .app-header {
            text-align: center;
            margin-bottom: 20px;
        }
        .app-title {
            font-size: 2.2rem;
            font-weight: 800;
            margin: 0;
            color: #1e293b;
        }
        .app-subtitle {
            color: #64748b;
            font-size: 1.05rem;
            margin-top: 4px;
        }
        .paper-preview-container {
            background: #f8fafc;
            border: 1px solid #e2e8f0;
            border-radius: 12px;
            padding: 16px;
            box-shadow: 0 4px 16px rgba(0,0,0,0.06);
            display: flex;
            flex-direction: column;
            align-items: center;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="app-header">
        <h1 class="app-title">✍️ Handwritten Page Generator</h1>
        <p class="app-subtitle">Type or paste full pages or paragraphs on the left to see realistic handwriting generated on the right</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Load model
model = load_model()
if model is None:
    st.error("⚠️ `generator.pt` was not found. Please upload it below:")
    uploaded = st.file_uploader("Upload generator.pt", type=["pt", "pth"])
    if uploaded is not None:
        g = Generator()
        g.load_state_dict(torch.load(uploaded, map_location="cpu"))
        g.eval()
        model = g
        st.success("✅ Model loaded successfully!")
    else:
        st.stop()

# Persistent state for seed
if "style_seed" not in st.session_state:
    st.session_state.style_seed = 42

DEFAULT_TEXT = (
    "Dear Professor,\n"
    "Here is my handwritten assignment for Deep Learning CA2.\n"
    "This model is a Conditional GAN trained on the EMNIST dataset.\n"
    "It can synthesize realistic handwriting for full pages and paragraphs seamlessly."
)

# -----------------------------------------------------------------------------
# SIDE-BY-SIDE 2-COLUMN LAYOUT
# -----------------------------------------------------------------------------
col_left, col_right = st.columns([5, 6], gap="large")

with col_left:
    st.markdown("### 📝 Enter Your Text")

    input_text = st.text_area(
        "Type or paste your text / essay / notes here:",
        value=DEFAULT_TEXT,
        height=280,
        placeholder="Type multiple lines, sentences, or paragraphs...",
        help="Type as many lines as you like. It automatically wraps onto the handwritten page.",
    )

    st.markdown("### 🎛️ Page & Text Controls")

    # 1. Text Size Slider (prominent and intuitive)
    char_size = st.slider(
        "📏 Text Size (Font Size)",
        min_value=20,
        max_value=56,
        value=32,
        step=2,
        help="Adjust the size of the handwritten characters on the page.",
    )

    # 2. Layout Controls
    ctrl_col1, ctrl_col2 = st.columns(2)
    with ctrl_col1:
        paper_style = st.selectbox(
            "📜 Paper Style",
            options=["Ruled Notebook", "Plain White", "Dark Canvas"],
            index=0,
            help="Choose between ruled notebook paper with margins or clean white paper.",
        )
        consistent_style = st.toggle(
            "Consistent Penmanship",
            value=True,
            help="Use uniform handwriting style across all characters.",
        )

    with ctrl_col2:
        ink_color = st.selectbox(
            "🖋️ Ink Color",
            options=["Classic Blue", "Dark Black"],
            index=0,
            help="Select handwriting ink color.",
        )
        max_chars = st.slider(
            "Page Width (Characters per line)",
            min_value=20,
            max_value=60,
            value=36,
            step=2,
            help="Controls word wrapping width per line.",
        )

    # 3. Style Seed / Regenerate Button
    seed_col1, seed_col2 = st.columns([3, 2])
    with seed_col1:
        st.session_state.style_seed = st.number_input(
            "Style Seed",
            min_value=0,
            max_value=999999,
            value=st.session_state.style_seed,
            step=1,
        )
    with seed_col2:
        st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
        if st.button("🎲 New Handwriting Style", use_container_width=True):
            st.session_state.style_seed = int(np.random.randint(0, 999999))
            st.rerun()

with col_right:
    st.markdown("### 📄 Handwritten Page Preview")

    if input_text.strip():
        # Process and wrap text
        lines, skipped = wrap_text_into_lines(input_text, max_chars_per_line=max_chars)

        # Render page
        page_image = render_handwritten_page(
            lines=lines,
            model=model,
            seed=st.session_state.style_seed,
            char_size=char_size,
            max_chars_per_line=max_chars,
            consistent=consistent_style,
            paper_style=paper_style,
            ink_color=ink_color,
        )

        # Display preview inside nice container
        st.image(page_image, use_container_width=True)

        # Download button
        buf = io.BytesIO()
        page_image.save(buf, format="PNG")
        st.download_button(
            label="📥 Download Handwritten Page (PNG)",
            data=buf.getvalue(),
            file_name=f"handwritten_page_seed_{st.session_state.style_seed}.png",
            mime="image/png",
            use_container_width=True,
        )

        # Notices
        caption_info = f"Lines: {len(lines)} | Text Size: {char_size}px | Style: {paper_style} ({ink_color})"
        if skipped:
            caption_info += f" | Note: Non-alphanumeric characters ({', '.join(skipped)}) were skipped."
        st.caption(caption_info)
    else:
        st.info("👈 Enter text in the box on the left to generate your handwritten page.")
