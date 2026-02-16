import { 
    VoxtralForConditionalGeneration, 
    VoxtralProcessor, 
    TextStreamer, 
    read_audio 
} from "@huggingface/transformers";

const status = document.getElementById('status');
const output = document.getElementById('output');
const button = document.getElementById('generate');

async function runAI() {
    status.textContent = "Initialisation du modèle Voxtral (chargement lourd)...";
    button.disabled = true;
    output.textContent = "";

    try {
        const model_id = "onnx-community/Voxtral-Mini-3B-2507-ONNX";

        // 1. Chargement du processeur (pour transformer le texte/audio en chiffres)
        const processor = await VoxtralProcessor.from_pretrained(model_id);

        // 2. Chargement du modèle avec les réglages optimisés (Quantization q4)
        // Note : On utilise 'q4' pour que ça tienne dans la mémoire de ta carte graphique
        const model = await VoxtralForConditionalGeneration.from_pretrained(model_id, {
            dtype: {
                embed_tokens: "fp16",
                audio_encoder: "q4", 
                decoder_model_merged: "q4",
            },
            device: "webgpu",
        });

        status.textContent = "Modèle prêt ! Test de transcription en cours...";

        // 3. Préparation du test (On utilise un fichier audio d'exemple)
        const audioUrl = "https://huggingface.co/datasets/Xenova/transformers.js-docs/resolve/main/jfk.wav";
        const audio = await read_audio(audioUrl, 16000);

        // Format spécial pour Voxtral (Chat Template)
        const conversation = [
            {
                role: "user",
                content: [
                    { type: "audio" },
                    { type: "text", text: "lang:en [TRANSCRIBE]" }, // Commande pour transcrire
                ],
            }
        ];

        const text = processor.apply_chat_template(conversation, { tokenize: false });
        const inputs = await processor(text, audio);

        // 4. Génération avec streaming (pour voir le texte s'afficher en direct)
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

        status.textContent = "Terminé !";

    } catch (error) {
        status.textContent = "Erreur : " + error.message;
        console.error(error);
    } finally {
        button.disabled = false;
    }
}

button.addEventListener('click', runAI);