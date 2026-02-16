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

    status.textContent = "Analyse de l'audio et génération...";
    generateBtn.disabled = true;
    output.textContent = "";
	
	// 1. Démarrer le chrono ICI (Dès le clic/début du traitement)
    const overallStartTime = performance.now(); 
    let firstTokenTime = null; // Pour calculer aussi la latence initiale si vous voulez
    let tokenCount = 0;

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
        
        const text = processor.apply_chat_template(conversation, { tokenize: false });
        const inputs = await processor(text, audioBuffer);

        const streamer = new TextStreamer(processor.tokenizer, {
            skip_special_tokens: true,
            skip_prompt: true,
            callback_function: (t) => {
                if (firstTokenTime === null) {
                    firstTokenTime = performance.now();
                    const latency = ((firstTokenTime - overallStartTime) / 1000).toFixed(2);
                    console.log(`Latence initiale (encodage audio) : ${latency}s`);
                }
                
                tokenCount++;
                output.textContent += t;

                const now = performance.now();
                // On calcule la vitesse sur la phase de génération pure
                const generationDuration = (now - firstTokenTime) / 1000;
                
                if (generationDuration > 0) {
                    const tps = (tokenCount / generationDuration).toFixed(2);
                    status.textContent = `Génération : ${tps} tokens/sec`;
                }
            }
        });

        await model.generate({
            ...inputs,
            max_new_tokens: 256,
            streamer,
        });

		const totalExecutionTime = ((performance.now() - overallStartTime) / 1000).toFixed(2);
        status.textContent = `Terminé en ${totalExecutionTime}s (Vitesse brute : ${(tokenCount / totalExecutionTime).toFixed(2)} t/s)`;
    } catch (error) {
        status.textContent = "Erreur génération : " + error.message;
        console.error(error);
    } finally {
        generateBtn.disabled = false;
    }
};