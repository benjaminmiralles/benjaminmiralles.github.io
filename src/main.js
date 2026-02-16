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

// On déclare les variables ici pour qu'elles soient globales au module
let model = null; 
let processor = null;
let mediaRecorder = null;
let audioChunks = [];
let audioBuffer = null;

async function initModel() {
    try {
        // 1. Vérification explicite du support WebGPU
        if (!navigator.gpu) {
            status.textContent = "WebGPU non supporté. Activez-le dans les réglages Safari.";
            return;
        }

        const model_id = "onnx-community/Voxtral-Mini-3B-2507-ONNX";
        
        // Configuration globale pour mobile
        status.textContent = "Initialisation du processeur...";
        
        processor = await VoxtralProcessor.from_pretrained(model_id);

        status.textContent = "Téléchargement des poids (0%)...";

        // 2. Chargement avec suivi de progression
        model = await VoxtralForConditionalGeneration.from_pretrained(model_id, {
            dtype: {
                embed_tokens: "fp32", 
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
        
        // Diagnostic spécifique iPad
        if (e.message.includes("out of memory") || e.message.includes("exhausted")) {
            status.textContent = "Erreur : Mémoire RAM saturée. Fermez les autres onglets.";
        }
    }
}

// Lancer l'initialisation immédiatement
initModel();

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
            
            // On n'active le bouton QUE si le modèle est bien chargé
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

        // --- AJOUT POUR LES STATISTIQUES ---
        let startTime = null;
        let tokenCount = 0;
        const statsDisplay = document.getElementById('status'); 
        // -----------------------------------

        const streamer = new TextStreamer(processor.tokenizer, {
            skip_special_tokens: true,
            skip_prompt: true,
            callback_function: (t) => {
                // On démarre le chrono au premier token reçu
                if (startTime === null) startTime = performance.now();
                
                tokenCount++;
                output.textContent += t;

                // Calcul de la vitesse
                const now = performance.now();
                const durationInSeconds = (now - startTime) / 1000;
                
                if (durationInSeconds > 0) {
                    const tps = (tokenCount / durationInSeconds).toFixed(2);
                    statsDisplay.textContent = `Transcription en cours : ${tps} tokens/sec`;
                }
            }
        });

        await model.generate({
            ...inputs,
            max_new_tokens: 256,
            streamer,
        });

        const totalTime = ((performance.now() - startTime) / 1000).toFixed(2);
        status.textContent = `Terminé en ${totalTime}s (${(tokenCount / totalTime).toFixed(2)} tokens/sec)`;

    } catch (error) {
        status.textContent = "Erreur génération : " + error.message;
        console.error(error);
    } finally {
        generateBtn.disabled = false;
    }
};