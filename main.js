const questions = [
    {
        "question": "sprachen",
        "answers": ["gesprochen", "gesprachen", "gespricht", "sprachen"],
        "correct": "gesprochen"
    },
    {
        "question": "machen",
        "answers": ["gemacht", "gemachen", "machen", "gemochen"],
        "correct": "gemacht"
    },
    {
        "question": "trinken",
        "answers": ["getrunken", "getrinken", "getrunkt", "getrankt"],
        "correct": "getrunken"
    }
];

let shuffledQuestions = [];
let currentQuestionIndex = 0;
let score = 0;
let totalQuestions = 0;

const startScreen = document.getElementById("start-screen");
const quizContainer = document.getElementById("quiz-container");
const resultScreen = document.getElementById("result-screen");
const loadingScreen = document.getElementById("loading-screen");

const questionElement = document.getElementById("question");
const answerButtons = document.querySelectorAll(".answer-btn");
const resultElement = document.getElementById("result");
const finalScoreElement = document.getElementById("final-score");

document.getElementById("start-btn").addEventListener("click", startQuiz);
document.getElementById("end-btn").addEventListener("click", endQuiz);
document.getElementById("restart-btn").addEventListener("click", restartQuiz);

function startQuiz() {
    startScreen.classList.add("hidden");
    loadingScreen.classList.remove("hidden");

    shuffledQuestions = shuffleArray(questions);
    currentQuestionIndex = 0;
    score = 0;
    totalQuestions = 0;

    // loadQuestion();

    /*fetch('questions.json')
        .then(response => response.json())
        .then(data => {
            shuffledQuestions = shuffleArray(data);
            loadQuestion();
        })
        .catch(error => console.error('Erreur de chargement des questions :', error));*/

        fetch('questions.json')
        .then(response => response.json())
        .then(data => {
            shuffledQuestions = shuffleArray(data);
            currentQuestionIndex = 0;
            score = 0;
            totalQuestions = 0;

            loadingScreen.classList.add("hidden"); // Cacher l'écran de chargement
            quizContainer.classList.remove("hidden"); // Afficher le quiz
            loadQuestion();
        })
        .catch(error => {
            console.error('Erreur de chargement des questions :', error);
            loadingScreen.innerHTML = '<h1>Erreur de chargement des questions</h1>';
        });
}

function loadQuestion() {
    const currentQuestion = shuffledQuestions[currentQuestionIndex];
    questionElement.textContent = currentQuestion.question;

    const shuffledAnswers = shuffleArray(currentQuestion.answers);
    answerButtons.forEach((button, index) => {
        button.textContent = shuffledAnswers[index];
        button.onclick = () => checkAnswer(button.textContent);
    });
}

function checkAnswer(selectedAnswer) {
    totalQuestions++;
    const currentQuestion = shuffledQuestions[currentQuestionIndex];
    if (selectedAnswer === currentQuestion.correct) {
        score++;
        resultElement.textContent = "Genau!";
        resultElement.className = "correct";
    } else {
        resultElement.textContent = "Schade!";
        resultElement.className = "incorrect";
    }

    resultElement.style.display = "block";
    resultElement.classList.add("show-result");

    setTimeout(() => {
        resultElement.style.display = "none";
        resultElement.classList.remove("show-result");
        currentQuestionIndex = (currentQuestionIndex + 1) % shuffledQuestions.length;
        loadQuestion();
    }, 1000);
}

function endQuiz() {
    quizContainer.classList.add("hidden");
    resultScreen.classList.remove("hidden");
    finalScoreElement.textContent = `Score: ${score}/${totalQuestions}`;
}

function restartQuiz() {
    resultScreen.classList.add("hidden");
    startScreen.classList.remove("hidden");
}

function shuffleArray(array) {
    for (let i = array.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [array[i], array[j]] = [array[j], array[i]];
    }
    return array;
}
