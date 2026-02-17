import {
    VoxtralForConditionalGeneration,
    VoxtralProcessor,
    TextStreamer
} from "@huggingface/transformers";

const status = document.getElementById('status');
const recordStatus = document.getElementById('recordStatus');
const output = document.getElementById('output');
const generateBtn = document.getElementById('generate');
const recordBtn = document.getElementById('recordBtn');
const progressContainer = document.getElementById('loadingProgress');
const progressLabel = document.getElementById('progressLabel');
const progressText = document.getElementById('progressText');
const progressBar = document.getElementById('progressBar');
const historyList = document.getElementById('historyList');
const emptyHistory = document.getElementById('emptyHistory');

let model = null;
let processor = null;
let mediaRecorder = null;
let audioChunks = [];
let audioBuffer = null;

const MAX_MODEL_FILES = 3;
const downloadedFiles = new Map();
const transcriptHistory = [];
let selectedHistoryId = null;

function updateProgressUI() {
    const loadedCount = Math.min(downloadedFiles.size, MAX_MODEL_FILES);
    const totalPercent = Array.from(downloadedFiles.values()).reduce((acc, val) => acc + val, 0);
    const normalizedPercent = Math.min(100, Math.round(totalPercent / MAX_MODEL_FILES));

    progressBar.style.width = `${normalizedPercent}%`;
    progressText.textContent = `${loadedCount} / ${MAX_MODEL_FILES}`;

    if (loadedCount >= MAX_MODEL_FILES) {
        progressLabel.textContent = 'Fichiers modèle téléchargés';
    }
}

function trackModelDownload(progressInfo) {
    if (!progressInfo || !progressInfo.file) {
        return;
    }

    if (progressInfo.status === 'done') {
        downloadedFiles.set(progressInfo.file, 100);
        updateProgressUI();
        return;
    }

    if (progressInfo.status === 'progress') {
        const currentPercent = progressInfo.progress ?? (
            progressInfo.total
                ? (progressInfo.loaded / progressInfo.total) * 100
                : 0
        );
        downloadedFiles.set(progressInfo.file, Math.max(0, Math.min(100, currentPercent)));
        updateProgressUI();
    }
}

function hideProgressIfReady() {
    progressBar.style.width = '100%';
    progressText.textContent = `${MAX_MODEL_FILES} / ${MAX_MODEL_FILES}`;
    progressLabel.textContent = 'Téléchargement terminé';
    setTimeout(() => {
        progressContainer.classList.add('hidden');
    }, 800);
}

function formatDate(timestamp) {
    return new Intl.DateTimeFormat('fr-FR', {
        dateStyle: 'short',
        timeStyle: 'medium',
    }).format(timestamp);
}

function renderHistory() {
    historyList.querySelectorAll('.history-item').forEach((item) => item.remove());

    if (transcriptHistory.length === 0) {
        emptyHistory.classList.remove('hidden');
        return;
    }

    emptyHistory.classList.add('hidden');

    transcriptHistory.forEach((entry) => {
        const button = document.createElement('button');
        button.type = 'button';
        button.className = `history-item${entry.id === selectedHistoryId ? ' active' : ''}`;
        button.dataset.id = String(entry.id);

        const title = document.createElement('span');
        title.className = 'history-title';
        title.textContent = entry.preview;

        const subtitle = document.createElement('span');
        subtitle.className = 'history-subtitle';
        subtitle.textContent = formatDate(entry.createdAt);

        button.append(title, subtitle);
        button.addEventListener('click', () => {
            selectedHistoryId = entry.id;
            output.textContent = entry.text;
            status.textContent = `Historique affiché (${formatDate(entry.createdAt)})`;
            renderHistory();
        });

        historyList.appendChild(button);
    });
}

function addTranscriptToHistory(text) {
    const cleanText = text.trim();
    if (!cleanText) {
        return;
    }

    const preview = cleanText.length > 40 ? `${cleanText.slice(0, 40)}…` : cleanText;
    const entry = {
        id: Date.now(),
        createdAt: new Date(),
        preview,
        text: cleanText,
    };

    transcriptHistory.unshift(entry);
    selectedHistoryId = entry.id;
    renderHistory();
}

async function initModel() {
    try {
        status.textContent = "Chargement du modèle Voxtral (WebGPU)...";
        recordBtn.disabled = true;

        const model_id = "onnx-community/Voxtral-Mini-3B-2507-ONNX";
        const progress_callback = (data) => trackModelDownload(data);

        processor = await VoxtralProcessor.from_pretrained(model_id, {
            progress_callback,
        });
        model = await VoxtralForConditionalGeneration.from_pretrained(model_id, {
            dtype: {
                embed_tokens: "fp16",
                audio_encoder: "q4",
                decoder_model_merged: "q4",
            },
            device: "webgpu",
            progress_callback,
        });

        hideProgressIfReady();
        status.textContent = "Modèle prêt ! Enregistrez un message.";
        recordBtn.disabled = false;
    } catch (e) {
        status.textContent = "Erreur de chargement : " + e.message;
        console.error("Erreur initModel:", e);
    }
}

initModel();
renderHistory();

recordBtn.onclick = async () => {
    if (mediaRecorder && mediaRecorder.state === "recording") {
        mediaRecorder.stop();
        recordBtn.textContent = "Démarrer l'enregistrement";
        recordStatus.innerHTML = "";
        return;
    }

    try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        mediaRecorder = new MediaRecorder(stream);
        audioChunks = [];

        mediaRecorder.ondataavailable = (e) => audioChunks.push(e.data);

        mediaRecorder.onstop = async () => {
            const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
            const arrayBuffer = await audioBlob.arrayBuffer();
            const audioContext = new AudioContext({ sampleRate: 16000 });
            const decoded = await audioContext.decodeAudioData(arrayBuffer);

            audioBuffer = decoded.getChannelData(0);

            if (model) {
                generateBtn.disabled = false;
                status.textContent = "Audio prêt. Cliquez sur Lancer.";
            } else {
                status.textContent = "Audio prêt, mais le modèle charge encore...";
            }
        };

        mediaRecorder.start();
        recordBtn.textContent = "Arrêter l'enregistrement";
        recordStatus.innerHTML = '<span class="recording">● Enregistrement en cours...</span>';
    } catch (err) {
        status.textContent = "Erreur micro : " + err.message;
    }
};

generateBtn.onclick = async () => {
    if (!model || !processor) {
        status.textContent = "Erreur : Le modèle n'est pas encore chargé.";
        return;
    }
    if (!audioBuffer) {
        status.textContent = "Erreur : Aucun audio enregistré.";
        return;
    }

    status.textContent = "Transcription en cours...";
    generateBtn.disabled = true;
    output.textContent = "";

    try {
        const conversation = [
            {
                role: "user",
                content: [
                    { type: "audio" },
                    {
                        type: "text",
                        text: "Transcris cet audio en français. Ajoute la ponctuation et corrige les hésitations (euh, ah). Sois très précis sur les termes techniques."
                    },
                ],
            }
        ];
        const text = processor.apply_chat_template(conversation, { tokenize: false });
        const inputs = await processor(text, audioBuffer);

        const streamer = new TextStreamer(processor.tokenizer, {
            skip_special_tokens: true,
            skip_prompt: true,
            callback_function: (t) => {
                output.textContent += t;
            }
        });

        await model.generate({
            ...inputs,
            max_new_tokens: 256,
            streamer,
        });

        addTranscriptToHistory(output.textContent);
        status.textContent = "Terminé !";
    } catch (error) {
        status.textContent = "Erreur génération : " + error.message;
        console.error(error);
    } finally {
        generateBtn.disabled = false;
    }
};
