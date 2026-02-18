import {
    VoxtralForConditionalGeneration,
    VoxtralProcessor,
    TextStreamer
} from "@huggingface/transformers";

const recordStatus = document.getElementById('recordStatus');
const output = document.getElementById('output');
const summaryOutput = document.getElementById('summaryOutput');
const summaryPromptInput = document.getElementById('summaryPrompt');
const generateBtn = document.getElementById('generate');
const summarizeBtn = document.getElementById('summarize');
const recordBtn = document.getElementById('recordBtn');
const progressContainer = document.getElementById('loadingProgress');
const progressLabel = document.getElementById('progressLabel');
const progressText = document.getElementById('progressText');
const progressBar = document.getElementById('progressBar');
const progressFiles = document.getElementById('progressFiles');
const throughputInfo = document.getElementById('throughputInfo');
const summaryThroughputInfo = document.getElementById('summaryThroughputInfo');
const throughputBar = document.querySelector('.throughput-bar');
const historyList = document.getElementById('historyList');
const emptyHistory = document.getElementById('emptyHistory');

let model = null;
let processor = null;
let mediaRecorder = null;
let audioChunks = [];
let audioBuffer = null;

const DEFAULT_SUMMARY_PROMPT = "Résume ma transcription. Ne mets aucun titre, aucun sous-titre, ni aucune puce. Juste le résumé. Produis uniquement un texte suivi (paragraphes narratifs). Commence directement le résumé sans écrire **Résumé** ou similaire. Génère un résumé structuré à partir de la transcription suivante :";
const HISTORY_STORAGE_KEY = 'voxtral-transcript-history';

const MODEL_FILES_TO_TRACK = [
    'embed_tokens_fp16.onnx_data',
    'audio_encoder_q4f16.onnx_data',
    'decoder_model_merged_q4f16.onnx_data',
];
const DEFAULT_MODEL_FILE_TOTALS = {
    'embed_tokens_fp16.onnx_data': 192 * 1024 * 1024,
    'audio_encoder_q4f16.onnx_data': 357 * 1024 * 1024,
    'decoder_model_merged_q4f16.onnx_data': 312 * 1024 * 1024,
};
const MAX_MODEL_FILES = MODEL_FILES_TO_TRACK.length;
const downloadedFiles = new Map(
    MODEL_FILES_TO_TRACK.map((fileName) => [fileName, {
        loaded: 0,
        total: 0,
        percent: 0,
    }])
);
const transcriptHistory = [];
let selectedHistoryId = null;

if (summaryPromptInput) {
    summaryPromptInput.value = DEFAULT_SUMMARY_PROMPT;
}

function normalizeHistoryEntry(rawEntry) {
    if (!rawEntry || typeof rawEntry !== 'object') {
        return null;
    }

    const text = typeof rawEntry.text === 'string' ? rawEntry.text.trim() : '';
    if (!text) {
        return null;
    }

    const id = Number.isFinite(Number(rawEntry.id)) ? Number(rawEntry.id) : Date.now();
    const createdAtDate = new Date(rawEntry.createdAt || Date.now());
    const createdAt = Number.isNaN(createdAtDate.getTime()) ? new Date() : createdAtDate;
    const preview = text.length > 40 ? `${text.slice(0, 40)}…` : text;

    return {
        id,
        createdAt,
        preview,
        text,
        summaryPrompt: typeof rawEntry.summaryPrompt === 'string' && rawEntry.summaryPrompt.trim()
            ? rawEntry.summaryPrompt
            : DEFAULT_SUMMARY_PROMPT,
        summary: typeof rawEntry.summary === 'string' ? rawEntry.summary : '',
    };
}

function loadHistoryFromStorage() {
    let parsedHistory = [];

    try {
        const raw = localStorage.getItem(HISTORY_STORAGE_KEY);
        if (!raw) {
            return;
        }

        const parsed = JSON.parse(raw);
        if (Array.isArray(parsed)) {
            parsedHistory = parsed;
        }
    } catch (error) {
        console.warn('Impossible de charger l\'historique local :', error);
        return;
    }

    const sanitizedHistory = parsedHistory
        .map(normalizeHistoryEntry)
        .filter(Boolean)
        .sort((a, b) => b.createdAt.getTime() - a.createdAt.getTime());

    transcriptHistory.splice(0, transcriptHistory.length, ...sanitizedHistory);
}

function saveHistoryToStorage() {
    const serializableHistory = transcriptHistory.map((entry) => ({
        id: entry.id,
        createdAt: entry.createdAt.toISOString(),
        text: entry.text,
        summaryPrompt: entry.summaryPrompt,
        summary: entry.summary,
    }));

    try {
        localStorage.setItem(HISTORY_STORAGE_KEY, JSON.stringify(serializableHistory));
    } catch (error) {
        console.warn('Impossible d\'enregistrer l\'historique local :', error);
    }
}

function updateThroughputBarVisibility() {
    const hasVisibleCounter = !throughputInfo.classList.contains('hidden')
        || !summaryThroughputInfo.classList.contains('hidden');

    throughputBar.classList.toggle('hidden', !hasVisibleCounter);
}

function formatMegabytes(value) {
    if (!Number.isFinite(value) || value < 0) {
        return '0.00 Mo';
    }

    return `${(value / (1024 * 1024)).toFixed(2)} Mo`;
}

function getFileName(path) {
    if (!path) {
        return 'Fichier inconnu';
    }

    const chunks = String(path).split('/');
    return chunks[chunks.length - 1] || String(path);
}

function resolveTrackedFile(path) {
    const rawFileName = getFileName(path).split('?')[0];

    if (MODEL_FILES_TO_TRACK.includes(rawFileName)) {
        return rawFileName;
    }

    return MODEL_FILES_TO_TRACK.find((trackedFileName) => rawFileName.includes(trackedFileName))
        || null;
}

function renderFileProgress() {
    progressFiles.innerHTML = '';

    Array.from(downloadedFiles.entries())
        .slice(0, MAX_MODEL_FILES)
        .forEach(([file, fileProgress]) => {
            const item = document.createElement('li');
            item.className = 'progress-file-item';

            const name = document.createElement('span');
            name.className = 'progress-file-name';
            name.textContent = getFileName(file);

            const size = document.createElement('span');
            size.className = 'progress-file-size';
            const loadedText = formatMegabytes(fileProgress.loaded);
            const totalText = fileProgress.total ? formatMegabytes(fileProgress.total) : '...';
            size.textContent = `${loadedText} / ${totalText}`;

            item.append(name, size);
            progressFiles.appendChild(item);
        });
}

function updateProgressUI() {
    const trackedFiles = Array.from(downloadedFiles.values()).slice(0, MAX_MODEL_FILES);
    const loadedCount = trackedFiles.filter((item) => item.percent >= 100).length;
    const totalPercent = trackedFiles.reduce((acc, val) => acc + val.percent, 0);
    const normalizedPercent = Math.min(100, Math.round(totalPercent / MAX_MODEL_FILES));

    progressBar.style.width = `${normalizedPercent}%`;
    progressText.textContent = `${loadedCount} / ${MAX_MODEL_FILES}`;

    if (loadedCount >= MAX_MODEL_FILES) {
        progressLabel.textContent = 'Fichiers modèle téléchargés';
    }

    renderFileProgress();
}

function trackModelDownload(progressInfo) {
    if (!progressInfo) {
        return;
    }

    const trackedFile = resolveTrackedFile(progressInfo.file);
    if (!trackedFile) {
        return;
    }

    const existingFileProgress = downloadedFiles.get(trackedFile) ?? {
        loaded: 0,
        total: progressInfo.total || DEFAULT_MODEL_FILE_TOTALS[trackedFile] || 0,
        percent: 0,
    };

    if (progressInfo.status === 'done') {
        const total = progressInfo.total
            ?? existingFileProgress.total
            ?? DEFAULT_MODEL_FILE_TOTALS[trackedFile]
            ?? 0;

        downloadedFiles.set(trackedFile, {
            loaded: total,
            total,
            percent: 100,
        });
        updateProgressUI();
        return;
    }

    if (progressInfo.status === 'progress') {
        const loaded = progressInfo.loaded ?? existingFileProgress.loaded;
        const total = progressInfo.total ?? existingFileProgress.total;
        const currentPercent = progressInfo.progress ?? (
            total
                ? (loaded / total) * 100
                : existingFileProgress.percent
        );

        downloadedFiles.set(trackedFile, {
            loaded,
            total,
            percent: Math.max(0, Math.min(100, currentPercent)),
        });
        updateProgressUI();
    }
}

function finalizeProgressUI() {
    MODEL_FILES_TO_TRACK.forEach((fileName) => {
        const current = downloadedFiles.get(fileName) ?? { loaded: 0, total: 0, percent: 0 };
        const total = current.total || DEFAULT_MODEL_FILE_TOTALS[fileName] || 0;
        downloadedFiles.set(fileName, {
            loaded: total,
            total,
            percent: 100,
        });
    });

    progressLabel.textContent = 'Téléchargement terminé';
    setTimeout(() => {
        progressContainer.classList.add('hidden');
    }, 1100);
}

function formatDate(timestamp) {
    return new Intl.DateTimeFormat('fr-FR', {
        dateStyle: 'short',
        timeStyle: 'medium',
    }).format(timestamp);
}

function getSelectedEntry() {
    return transcriptHistory.find((entry) => entry.id === selectedHistoryId) ?? null;
}

function applyEntryToView(entry) {
    if (!entry) {
        output.textContent = '';
        summaryOutput.textContent = '';
        if (summaryPromptInput) {
            summaryPromptInput.value = DEFAULT_SUMMARY_PROMPT;
        }
        summarizeBtn.disabled = true;
        return;
    }

    output.textContent = entry.text;
    summaryOutput.textContent = entry.summary || '';
    if (summaryPromptInput) {
        summaryPromptInput.value = entry.summaryPrompt || DEFAULT_SUMMARY_PROMPT;
    }
    summarizeBtn.disabled = false;
}

function renderHistory() {
    historyList.querySelectorAll('.history-row').forEach((item) => item.remove());

    if (transcriptHistory.length === 0) {
        emptyHistory.classList.remove('hidden');
        return;
    }

    emptyHistory.classList.add('hidden');

    transcriptHistory.forEach((entry) => {
        const row = document.createElement('div');
        row.className = 'history-row';

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
            applyEntryToView(entry);
            renderHistory();
        });

        const deleteButton = document.createElement('button');
        deleteButton.type = 'button';
        deleteButton.className = 'history-delete';
        deleteButton.setAttribute('aria-label', 'Supprimer cette transcription');
        deleteButton.textContent = 'Supprimer';
        deleteButton.addEventListener('click', () => {
            deleteHistoryEntry(entry.id);
        });

        row.append(button, deleteButton);
        historyList.appendChild(row);
    });
}

function updateSelectedEntry(patch) {
    const entry = getSelectedEntry();
    if (!entry) {
        return;
    }

    Object.assign(entry, patch);
    if (typeof entry.text === 'string') {
        const cleanText = entry.text.trim();
        entry.preview = cleanText.length > 40 ? `${cleanText.slice(0, 40)}…` : cleanText;
    }

    saveHistoryToStorage();
    renderHistory();
}

function deleteHistoryEntry(entryId) {
    const index = transcriptHistory.findIndex((entry) => entry.id === entryId);
    if (index === -1) {
        return;
    }

    transcriptHistory.splice(index, 1);

    if (selectedHistoryId === entryId) {
        selectedHistoryId = transcriptHistory[0]?.id ?? null;
        applyEntryToView(getSelectedEntry());
    }

    saveHistoryToStorage();
    renderHistory();
}

function addTranscriptToHistory(text) {
    const cleanText = text.trim();
    if (!cleanText) {
        return;
    }

    const preview = cleanText.length > 40 ? `${cleanText.slice(0, 40)}…` : cleanText;
    const entry = {
        id: Date.now() + Math.floor(Math.random() * 1000),
        createdAt: new Date(),
        preview,
        text: cleanText,
        summaryPrompt: summaryPromptInput?.value?.trim() || DEFAULT_SUMMARY_PROMPT,
        summary: '',
    };

    transcriptHistory.unshift(entry);
    selectedHistoryId = entry.id;
    saveHistoryToStorage();
    renderHistory();
}

async function initModel() {
    try {
        recordBtn.disabled = true;

        const model_id = "onnx-community/Voxtral-Mini-3B-2507-ONNX";
        const progress_callback = (data) => trackModelDownload(data);

        processor = await VoxtralProcessor.from_pretrained(model_id, {
            progress_callback,
        });

        model = await VoxtralForConditionalGeneration.from_pretrained(model_id, {
            dtype: {
                embed_tokens: "fp16",
                audio_encoder: "q4f16",
                decoder_model_merged: "q4f16",
            },
            device: "webgpu",
            progress_callback,
        });

        finalizeProgressUI();
        recordBtn.disabled = false;
    } catch (e) {
        recordStatus.textContent = "Erreur de chargement : " + e.message;
        console.error("Erreur initModel:", e);
    }
}

loadHistoryFromStorage();
if (transcriptHistory.length > 0) {
    selectedHistoryId = transcriptHistory[0].id;
    applyEntryToView(transcriptHistory[0]);
}
updateProgressUI();
initModel();
renderHistory();
updateThroughputBarVisibility();

summaryPromptInput?.addEventListener('input', () => {
    const value = summaryPromptInput.value || DEFAULT_SUMMARY_PROMPT;
    updateSelectedEntry({ summaryPrompt: value });
});

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
            summarizeBtn.disabled = true;
            summaryOutput.textContent = '';

            if (model) {
                generateBtn.disabled = false;
            }
        };

        mediaRecorder.start();
        recordBtn.textContent = "Arrêter l'enregistrement";
        recordStatus.innerHTML = '<span class="recording">● Enregistrement en cours...</span>';
    } catch (err) {
        recordStatus.textContent = "Erreur micro : " + err.message;
    }
};

generateBtn.onclick = async () => {
    if (!model || !processor) {
        recordStatus.textContent = "Erreur : Le modèle n'est pas encore chargé.";
        return;
    }
    if (!audioBuffer) {
        recordStatus.textContent = "Erreur : Aucun audio enregistré.";
        return;
    }

    generateBtn.disabled = true;
    summarizeBtn.disabled = true;
    output.textContent = "";
    throughputInfo.classList.add('hidden');
    throughputInfo.textContent = '';
    summaryThroughputInfo.classList.add('hidden');
    summaryThroughputInfo.textContent = '';
    updateThroughputBarVisibility();

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

        const generationStart = performance.now();

        await model.generate({
            ...inputs,
            max_new_tokens: 512,
            streamer,
        });

        const generationDurationSeconds = (performance.now() - generationStart) / 1000;
        const tokenIds = processor.tokenizer.encode(output.textContent, { add_special_tokens: false });
        const generatedTokens = tokenIds.length;
        const tokensPerSecond = generationDurationSeconds > 0
            ? generatedTokens / generationDurationSeconds
            : 0;

        throughputInfo.textContent = `Débit transcript : ${tokensPerSecond.toFixed(2)} tokens/s (${generatedTokens} tokens)`;
        throughputInfo.classList.remove('hidden');
        updateThroughputBarVisibility();

        addTranscriptToHistory(output.textContent);
        summarizeBtn.disabled = false;
    } catch (error) {
        recordStatus.textContent = "Erreur génération : " + error.message;
        console.error(error);
    } finally {
        generateBtn.disabled = false;
    }
};

summarizeBtn.onclick = async () => {
    if (!model || !processor) {
        recordStatus.textContent = "Erreur : Le modèle n'est pas encore chargé.";
        return;
    }

    const transcriptText = output.textContent.trim();
    if (!transcriptText) {
        recordStatus.textContent = "Erreur : Aucun transcript à résumer.";
        return;
    }

    summarizeBtn.disabled = true;
    summaryOutput.textContent = '';
    summaryThroughputInfo.classList.add('hidden');
    summaryThroughputInfo.textContent = '';
    updateThroughputBarVisibility();

    try {
        const summaryPrompt = summaryPromptInput?.value?.trim() || DEFAULT_SUMMARY_PROMPT;
        const conversation2 = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": summaryPrompt + transcriptText
                    },
                ],
            }
        ];
        const text2 = processor.apply_chat_template(conversation2, { tokenize: false });
        const inputs2 = await processor(text2);

        const streamer = new TextStreamer(processor.tokenizer, {
            skip_special_tokens: true,
            skip_prompt: true,
            callback_function: (t) => {
                summaryOutput.textContent += t;
            }
        });

        const generationStart = performance.now();

        await model.generate({
            ...inputs2,
            max_new_tokens: 512,
            streamer,
        });

        const generationDurationSeconds = (performance.now() - generationStart) / 1000;
        const tokenIds = processor.tokenizer.encode(summaryOutput.textContent, { add_special_tokens: false });
        const generatedTokens = tokenIds.length;
        const tokensPerSecond = generationDurationSeconds > 0
            ? generatedTokens / generationDurationSeconds
            : 0;

        summaryThroughputInfo.textContent = `Débit résumé : ${tokensPerSecond.toFixed(2)} tokens/s (${generatedTokens} tokens)`;
        summaryThroughputInfo.classList.remove('hidden');
        updateThroughputBarVisibility();

        updateSelectedEntry({
            text: transcriptText,
            summaryPrompt,
            summary: summaryOutput.textContent,
        });
    } catch (error) {
        recordStatus.textContent = "Erreur génération : " + error.message;
        console.error(error);
    } finally {
        summarizeBtn.disabled = false;
    }
};
