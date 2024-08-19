const questions = [
    {
        "question": "sprachen",
        "answers": ["to do", "to speak", "to write", "to run"],
        "correct": "to speak"
    },
    {
        "question": "machen",
        "answers": ["to make", "to take", "to see", "to go"],
        "correct": "to make"
    }
    // Ajoutez d'autres questions ici
];

let currentQuestionIndex = 0;
let score = 0;
let totalQuestions = 0;

const startScreen = document.getElementById("start-screen");
const quizContainer = document.getElementById("quiz-container");
const resultScreen = document.getElementById("result-screen");

const questionElement = document.getElementById("question");
const answerButtons = document.querySelectorAll(".answer-btn");
const resultElement = document.getElementById("result");
const finalScoreElement = document.getElementById("final-score");

document.getElementById("start-btn").addEventListener("click", startQuiz);
document.getElementById("end-btn").addEventListener("click", endQuiz);
document.getElementById("restart-btn").addEventListener("click", restartQuiz);

function startQuiz() {
    startScreen.classList.add("hidden");
    quizContainer.classList.remove("hidden");
    loadQuestion();
}

function loadQuestion() {
    const currentQuestion = questions[currentQuestionIndex];
    questionElement.textContent = currentQuestion.question;
    answerButtons.forEach((button, index) => {
        button.textContent = currentQuestion.answers[index];
        button.onclick = () => checkAnswer(button.textContent);
    });
}

function checkAnswer(selectedAnswer) {
    totalQuestions++;
    const currentQuestion = questions[currentQuestionIndex];
    if (selectedAnswer === currentQuestion.correct) {
        score++;
        resultElement.textContent = "Correct!";
    } else {
        resultElement.textContent = "Incorrect!";
    }

    setTimeout(() => {
        resultElement.textContent = "";
        currentQuestionIndex = (currentQuestionIndex + 1) % questions.length;
        loadQuestion();
    }, 1000);
}

function endQuiz() {
    quizContainer.classList.add("hidden");
    resultScreen.classList.remove("hidden");
    finalScoreElement.textContent = `Score: ${score}/${totalQuestions}`;
}

function restartQuiz() {
    score = 0;
    totalQuestions = 0;
    currentQuestionIndex = 0;
    resultScreen.classList.add("hidden");
    startScreen.classList.remove("hidden");
}
