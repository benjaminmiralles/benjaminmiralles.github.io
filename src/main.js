import {
    VoxtralForConditionalGeneration,
    VoxtralProcessor,
    TextStreamer
} from "@huggingface/transformers";

const recordStatus = document.getElementById('recordStatus');
const output = document.getElementById('output');
const summaryOutput = document.getElementById('summaryOutput');
const generateBtn = document.getElementById('generate');
const summarizeBtn = document.getElementById('summarize');
const recordBtn = document.getElementById('recordBtn');
const progressContainer = document.getElementById('loadingProgress');
const progressLabel = document.getElementById('progressLabel');
const progressText = document.getElementById('progressText');
const progressBar = document.getElementById('progressBar');
const progressFiles = document.getElementById('progressFiles');
const throughputInfo = document.getElementById('throughputInfo');
const historyList = document.getElementById('historyList');
const emptyHistory = document.getElementById('emptyHistory');

let model = null;
let processor = null;
let mediaRecorder = null;
let audioChunks = [];
let audioBuffer = null;

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
            summarizeBtn.disabled = false;
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

updateProgressUI();
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
            max_new_tokens: 256,
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

    try {
        const conversation2 = [
			{
				"role": "user",
				"content": [
					{
						"type": "text",
						"text": "Agis comme un assistant expert en rédaction de comptes-rendus. Ne mets aucun titre, aucun sous-titre, ni aucune puce. Juste le résumé. Produis uniquement un texte suivi (paragraphes narratifs). Génère un résumé structuré à partir de la transcription suivante :" + transcriptText
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

        await model.generate({
            ...inputs2,
            max_new_tokens: 256,
            streamer,
        });
		
    } catch (error) {
        recordStatus.textContent = "Erreur génération : " + error.message;
        console.error(error);
    } finally {
        summarizeBtn.disabled = false;
    }
};
