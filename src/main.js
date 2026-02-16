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
const historyList = document.getElementById('historyList');
const progressWrapper = document.getElementById('progressWrapper');
const progressBar = document.getElementById('progressBar');
const progressValue = document.getElementById('progressValue');
const progressTrack = progressWrapper.querySelector('.progress-track');

const HISTORY_STORAGE_KEY = 'voxtral-transcripts-history';

let model = null;
let processor = null;
let mediaRecorder = null;
let audioChunks = [];
let audioBuffer = null;
let transcriptHistory = loadTranscriptHistory();
let selectedTranscriptId = transcriptHistory[0]?.id ?? null;

function loadTranscriptHistory() {
    try {
        const raw = localStorage.getItem(HISTORY_STORAGE_KEY);
        const parsed = raw ? JSON.parse(raw) : [];
        return Array.isArray(parsed) ? parsed : [];
    } catch {
        return [];
    }
}

function saveTranscriptHistory() {
    localStorage.setItem(HISTORY_STORAGE_KEY, JSON.stringify(transcriptHistory));
}

function renderHistory() {
    historyList.innerHTML = '';

    if (transcriptHistory.length === 0) {
        historyList.innerHTML = '<p class="empty-history">Aucun transcript pour le moment.</p>';
        return;
    }

    transcriptHistory.forEach((item, index) => {
        const button = document.createElement('button');
        button.className = `history-item ${item.id === selectedTranscriptId ? 'active' : ''}`;
        button.type = 'button';

        const date = new Date(item.createdAt).toLocaleString('fr-FR');

        button.innerHTML = `
            <span class="history-title">Transcript #${transcriptHistory.length - index}</span>
            <span class="history-subtitle">${date} · ${item.duration}s</span>
        `;

        button.onclick = () => {
            selectedTranscriptId = item.id;
            output.textContent = item.text;
            status.textContent = `Transcript chargé (${date})`;
            renderHistory();
        };

        historyList.appendChild(button);
    });
}

function addTranscriptToHistory(text, duration) {
    if (!text || !text.trim()) {
        return;
    }

    const entry = {
        id: crypto.randomUUID(),
        text: text.trim(),
        duration,
        createdAt: new Date().toISOString(),
    };

    transcriptHistory.unshift(entry);
    selectedTranscriptId = entry.id;

    saveTranscriptHistory();
    renderHistory();
}

function setProgress(percent, label) {
    const value = Math.max(0, Math.min(100, Math.round(percent)));
    progressWrapper.classList.remove('hidden');
    progressBar.style.width = `${value}%`;
    progressValue.textContent = `${value}%`;
    progressTrack.setAttribute('aria-valuenow', String(value));

    if (label) {
        status.textContent = label;
    }
}

function hideProgress() {
    progressWrapper.classList.add('hidden');
    progressBar.style.width = '0%';
    progressValue.textContent = '0%';
    progressTrack.setAttribute('aria-valuenow', '0');
}

async function initModel() {
    try {
        if (!navigator.gpu) {
            status.textContent = "WebGPU non supporté. Activez-le dans les réglages Safari.";
            return;
        }

        const model_id = "onnx-community/Voxtral-Mini-3B-2507-ONNX";

        status.textContent = "Initialisation du processeur...";
        processor = await VoxtralProcessor.from_pretrained(model_id);

        status.textContent = "Téléchargement des poids (0%)...";

        model = await VoxtralForConditionalGeneration.from_pretrained(model_id, {
            dtype: {
                embed_tokens: "q4",
                audio_encoder: "q4f16",
                decoder_model_merged: "q4f16",
            },
            device: "webgpu",
            progress_callback: (data) => {
                if (data.status === 'progress') {
                    status.textContent = `Téléchargement : ${data.file} (${Math.round(data.loaded / 1024 / 1024)} Mo)`;
                } else if (data.status === 'done') {
                    status.textContent = `Fichier chargé : ${data.file}`;
                }
            }
        });

        status.textContent = "Modèle prêt ! Enregistrez un message.";
        recordBtn.disabled = false;

    } catch (e) {
        console.error("Erreur complète :", e);
        status.textContent = "Erreur : " + e.message;

        if (e.message.includes("out of memory") || e.message.includes("exhausted")) {
            status.textContent = "Erreur : Mémoire RAM saturée. Fermez les autres onglets.";
        }
    }
}

function extractTextFromGenerationResult(generationResult, promptText) {
    try {
        const rawSequences = generationResult?.sequences ?? generationResult;
        if (!rawSequences) {
            return '';
        }

        const decoded = processor.tokenizer.batch_decode(rawSequences, {
            skip_special_tokens: true,
        });

        const first = decoded?.[0] ?? '';
        if (!first) {
            return '';
        }

        return first.replace(promptText, '').trim();
    } catch (error) {
        console.warn('Impossible de décoder la génération en fallback:', error);
        return '';
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

    generateBtn.disabled = true;
    output.textContent = "";

    const overallStartTime = performance.now();
    let firstTokenTime = null;
    let tokenCount = 0;
    let streamedText = '';

    setProgress(8, 'Préparation de la transcription...');

    setProgress(8, 'Préparation de la transcription...');

    try {
        const conversation = [
            {
                role: "user",
                content: [
                    { type: "audio" },
                    {
                        type: "text",
                        text: "Transcris cet audio en français. Ajoute la ponctuation et corrige les hésitations. Sois très précis."
                    },
                ],
            }
        ];

        const promptText = processor.apply_chat_template(conversation, { tokenize: false });
        setProgress(18, 'Encodage du prompt...');

        const inputs = await processor(promptText, audioBuffer);
        setProgress(30, 'Analyse de l\'audio...');

        const onToken = (t) => {
            if (firstTokenTime === null) {
                firstTokenTime = performance.now();
                const latency = ((firstTokenTime - overallStartTime) / 1000).toFixed(2);
                console.log(`Latence initiale (encodage audio) : ${latency}s`);
            }

            tokenCount++;
            streamedText += t;
            output.textContent = streamedText;

            const generationDuration = (performance.now() - firstTokenTime) / 1000;
            if (generationDuration > 0) {
                const tps = (tokenCount / generationDuration).toFixed(2);
                const estimatedProgress = Math.min(95, 35 + tokenCount * 2.5);
                setProgress(estimatedProgress, `Génération : ${tps} tokens/sec`);
            }
        };

        const streamer = new TextStreamer(processor.tokenizer, {
            skip_special_tokens: true,
            skip_prompt: true,
            callback_function: onToken,
            callbackFunction: onToken,
        });

        const generationResult = await model.generate({
            ...inputs,
            max_new_tokens: 256,
            streamer,
            return_dict_in_generate: true,
        });

        if (!streamedText.trim()) {
            const fallbackText = extractTextFromGenerationResult(generationResult, promptText);
            if (fallbackText) {
                streamedText = fallbackText;
                output.textContent = streamedText;
            }
        }

        if (!streamedText.trim()) {
            status.textContent = 'Transcription terminée mais vide. Réessayez avec un enregistrement plus long.';
            return;
        }

        const totalExecutionTime = ((performance.now() - overallStartTime) / 1000).toFixed(2);
        setProgress(100, 'Finalisation...');
        status.textContent = `Terminé en ${totalExecutionTime}s (Vitesse brute : ${(tokenCount / Number(totalExecutionTime || 1)).toFixed(2)} t/s)`;

        addTranscriptToHistory(streamedText, totalExecutionTime);
    } catch (error) {
        status.textContent = "Erreur génération : " + error.message;
        console.error(error);
    } finally {
        setTimeout(hideProgress, 500);
        generateBtn.disabled = false;
    }
};
