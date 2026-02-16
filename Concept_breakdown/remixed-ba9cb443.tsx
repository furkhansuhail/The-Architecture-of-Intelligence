import React, { useState } from 'react';

const TriviaGame = () => {
  const [gameState, setGameState] = useState('setup');
  const [categories, setCategories] = useState([]);
  const [difficulty, setDifficulty] = useState('medium');
  const [numQuestions, setNumQuestions] = useState(5);
  const [questions, setQuestions] = useState([]);
  const [currentQuestion, setCurrentQuestion] = useState(0);
  const [selectedAnswer, setSelectedAnswer] = useState(null);
  const [score, setScore] = useState(0);
  const [answers, setAnswers] = useState([]);
  const [showAnswer, setShowAnswer] = useState(false);

  const availableCategories = [
    { id: 'convolution', label: 'Convolution Operation', icon: '🔲' },
    { id: 'filters', label: 'Filters & Feature Maps', icon: '🔍' },
    { id: 'pooling', label: 'Pooling Layers', icon: '📉' },
    { id: 'architecture', label: 'CNN Architectures', icon: '🏗️' },
    { id: 'training', label: 'Training & Backprop', icon: '⚙️' },
    { id: 'regularization', label: 'Regularization', icon: '🛡️' },
    { id: 'batchnorm', label: 'Batch Normalization', icon: '📊' },
    { id: 'transfer', label: 'Transfer Learning', icon: '🔄' },
    { id: 'history', label: 'CNN History & Evolution', icon: '📜' },
    { id: 'transformers', label: 'CNNs vs Transformers', icon: '⚡' },
  ];

  const toggleCategory = (id) => {
    setCategories(prev =>
      prev.includes(id) ? prev.filter(c => c !== id) : [...prev, id]
    );
  };

  const generateQuestions = async () => {
    if (categories.length === 0) {
      alert('Please select at least one topic!');
      return;
    }
    setGameState('loading');

    const topicDescriptions = {
      convolution: "the convolution operation: how filters slide across images, dot products, element-wise multiply-and-sum, stride, padding, output size formulas, weight sharing, translation equivariance, multi-channel convolutions (RGB), 1x1 convolutions, and the difference between convolution and fully connected layers",
      filters: "CNN filters/kernels and feature maps: what different filters detect (edges, corners, textures), how multiple filters produce multiple feature maps, hierarchical feature learning (edges→textures→parts→objects), receptive fields, and how deeper layers see larger regions of the input",
      pooling: "pooling layers: max pooling, average pooling, global average pooling, stride in pooling, how pooling provides translation invariance, why max pooling preserves strong signals better than average pooling, zero learnable parameters in pooling, and backpropagation through pooling (gradient routing)",
      architecture: "CNN architecture design: LeNet-5, AlexNet, VGGNet, GoogLeNet/Inception modules, ResNet and skip/residual connections, bottleneck blocks, the degradation problem, how 1x1 convolutions enable bottleneck design, and the typical Conv→BatchNorm→ReLU→Pool pipeline",
      training: "training CNNs: forward pass, loss calculation (cross-entropy), backpropagation through conv layers, how gradients flow through shared weights, gradient accumulation across positions, learning rate, optimizers (Adam, SGD with momentum), learning rate schedulers",
      regularization: "regularization in CNNs: dropout (randomly zeroing neurons, co-adaptation prevention, dropout rates for FC vs conv layers, inference vs training behavior), data augmentation (flips, crops, color jitter), weight decay/L2 regularization, early stopping, overfitting vs underfitting",
      batchnorm: "batch normalization: internal covariate shift problem, the 4-step BatchNorm formula (mean, variance, normalize, scale+shift), learnable gamma and beta parameters, where BatchNorm sits in the layer order, benefits (higher learning rates, acts as regularizer, reduces initialization sensitivity)",
      transfer: "transfer learning: why early CNN layers are universal (edges, textures), feature extraction strategy (freeze+replace head), fine-tuning strategy (unfreeze later layers with small lr), when transfer learning breaks down, practical recipes for different dataset sizes",
      history: "CNN history and evolution: LeNet-5 (1998), AlexNet (2012) and the deep learning big bang, VGGNet's depth-with-small-filters insight, GoogLeNet's Inception modules and parallel multi-scale filters, ResNet's skip connections solving the degradation problem, parameter count trends across architectures",
      transformers: "CNNs vs Vision Transformers: spatial inductive biases in CNNs (locality, translation equivariance, hierarchy), how ViT splits images into patches and uses self-attention, data efficiency tradeoffs, hybrid architectures (ConvNeXt, CoAtNet), when CNNs remain the right choice"
    };

    const selectedTopics = categories.map(c => topicDescriptions[c]).join('; ');

    try {
      const prompt = `You are a deep learning instructor creating a quiz on Convolutional Neural Networks (CNNs).

Generate exactly ${numQuestions} multiple-choice trivia questions about CNN theory.

TOPICS TO COVER (focus on these): ${selectedTopics}

DIFFICULTY: ${difficulty}
- easy: definitions, basic concepts, identify what a component does
- medium: numerical reasoning (e.g. compute output sizes, parameter counts), compare approaches, explain why something works
- hard: subtle tradeoffs, edge cases, connecting multiple concepts, architectural design decisions

RULES:
- Questions must be technically accurate and precise
- Each question should have exactly 4 options with only ONE correct answer
- Wrong options should be plausible (common misconceptions or related concepts)
- Vary question types: some conceptual, some computational, some "which architecture introduced X?"
- For computational questions, use small concrete numbers so mental math is possible

Respond ONLY with valid JSON, no markdown, no preamble:
{
  "questions": [
    {
      "question": "The question text",
      "options": ["Option A", "Option B", "Option C", "Option D"],
      "correctAnswer": 0,
      "explanation": "Brief explanation of why the correct answer is right",
      "category": "Topic area"
    }
  ]
}`;

      const response = await window.claude.complete(prompt);
      const parsed = JSON.parse(response);
      if (parsed.questions && Array.isArray(parsed.questions)) {
        setQuestions(parsed.questions);
        setGameState('playing');
        setCurrentQuestion(0);
        setScore(0);
        setAnswers([]);
      } else {
        throw new Error('Bad format');
      }
    } catch (e) {
      console.error(e);
      alert('Failed to generate questions. Please try again.');
      setGameState('setup');
    }
  };

  const checkAnswer = () => {
    if (selectedAnswer === null) { alert('Please select an answer!'); return; }
    if (!showAnswer) { setShowAnswer(true); return; }
    const isCorrect = selectedAnswer === questions[currentQuestion].correctAnswer;
    const newAnswers = [...answers, { questionIndex: currentQuestion, selectedAnswer, isCorrect }];
    setAnswers(newAnswers);
    if (isCorrect) setScore(score + 1);
    if (currentQuestion + 1 < questions.length) {
      setCurrentQuestion(currentQuestion + 1);
      setSelectedAnswer(null);
      setShowAnswer(false);
    } else {
      setGameState('results');
    }
  };

  const resetGame = () => {
    setGameState('setup');
    setCurrentQuestion(0);
    setSelectedAnswer(null);
    setScore(0);
    setAnswers([]);
    setQuestions([]);
    setShowAnswer(false);
  };

  // ── SETUP ──
  if (gameState === 'setup') {
    return (
      <div className="min-h-screen bg-slate-950 text-white p-4">
        <div className="max-w-2xl mx-auto">
          <div className="text-center mt-8 mb-8">
            <div className="text-5xl mb-3">🧠</div>
            <h1 className="text-5xl font-black mb-2" style={{ color: '#60a5fa' }}>CNN Trivia</h1>
            <p className="text-lg text-slate-400">Test your knowledge of Convolutional Neural Networks</p>
          </div>
          <div className="rounded-2xl p-8 border-2" style={{ backgroundColor: '#1e293b', borderColor: '#334155' }}>
            <div className="mb-8">
              <h2 className="text-xl font-bold mb-4" style={{ color: '#60a5fa' }}>Select Topics</h2>
              <div className="grid grid-cols-2 gap-3">
                {availableCategories.map(cat => (
                  <button
                    key={cat.id}
                    onClick={() => toggleCategory(cat.id)}
                    className="p-3 rounded-xl font-semibold transition-all duration-200 border-2 text-left text-sm"
                    style={{
                      backgroundColor: categories.includes(cat.id) ? '#3b82f6' : '#334155',
                      color: categories.includes(cat.id) ? '#000' : '#fff',
                      borderColor: categories.includes(cat.id) ? '#60a5fa' : '#475569',
                    }}
                  >
                    <span className="mr-2">{cat.icon}</span>{cat.label}
                  </button>
                ))}
              </div>
            </div>
            <div className="mb-8">
              <h2 className="text-xl font-bold mb-4" style={{ color: '#60a5fa' }}>Difficulty</h2>
              <div className="flex gap-3">
                {['easy', 'medium', 'hard'].map(d => (
                  <button
                    key={d}
                    onClick={() => setDifficulty(d)}
                    className="px-6 py-3 rounded-xl font-semibold transition-all duration-200 border-2 capitalize"
                    style={{
                      backgroundColor: difficulty === d ? '#3b82f6' : '#334155',
                      color: difficulty === d ? '#000' : '#fff',
                      borderColor: difficulty === d ? '#60a5fa' : '#475569',
                    }}
                  >
                    {d}
                  </button>
                ))}
              </div>
            </div>
            <div className="mb-8">
              <h2 className="text-xl font-bold mb-4" style={{ color: '#60a5fa' }}>Questions</h2>
              <div className="flex gap-3">
                {[5, 10, 15].map(n => (
                  <button
                    key={n}
                    onClick={() => setNumQuestions(n)}
                    className="px-6 py-3 rounded-xl font-semibold transition-all duration-200 border-2"
                    style={{
                      backgroundColor: numQuestions === n ? '#3b82f6' : '#334155',
                      color: numQuestions === n ? '#000' : '#fff',
                      borderColor: numQuestions === n ? '#60a5fa' : '#475569',
                    }}
                  >
                    {n}
                  </button>
                ))}
              </div>
            </div>
            <button
              onClick={generateQuestions}
              className="w-full font-black py-4 px-8 rounded-xl text-xl transition-all duration-200"
              style={{ backgroundColor: '#3b82f6', color: '#000' }}
            >
              Start Quiz
            </button>
          </div>
        </div>
      </div>
    );
  }

  // ── LOADING ──
  if (gameState === 'loading') {
    return (
      <div className="min-h-screen bg-slate-950 text-white flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin w-16 h-16 border-4 rounded-full mx-auto mb-4" style={{ borderColor: '#60a5fa', borderTopColor: 'transparent' }}></div>
          <h2 className="text-2xl font-bold" style={{ color: '#60a5fa' }}>Generating CNN questions...</h2>
          <p className="text-slate-400 mt-2">Building your quiz</p>
        </div>
      </div>
    );
  }

  // ── PLAYING ──
  if (gameState === 'playing') {
    const q = questions[currentQuestion];
    return (
      <div className="min-h-screen bg-slate-950 text-white p-4">
        <div className="max-w-3xl mx-auto">
          <div className="flex justify-between items-center mb-6">
            <div className="font-bold text-lg" style={{ color: '#60a5fa' }}>
              Question {currentQuestion + 1} of {questions.length}
            </div>
            <div className="font-bold text-lg" style={{ color: '#60a5fa' }}>
              Score: {score}
            </div>
          </div>
          <div className="w-full rounded-full h-2 mb-6" style={{ backgroundColor: '#334155' }}>
            <div className="h-2 rounded-full transition-all duration-500" style={{ backgroundColor: '#3b82f6', width: `${((currentQuestion) / questions.length) * 100}%` }}></div>
          </div>
          <div className="rounded-2xl p-8 border-2" style={{ backgroundColor: '#1e293b', borderColor: '#334155' }}>
            {q.category && (
              <span className="inline-block px-3 py-1 rounded-full text-xs font-semibold mb-4" style={{ backgroundColor: '#3b82f6', color: '#000' }}>
                {q.category}
              </span>
            )}
            <h2 className="text-xl font-bold leading-relaxed mb-6">{q.question}</h2>
            <div className="space-y-3 mb-6">
              {q.options.map((opt, i) => {
                let bg = '#334155', border = '#475569', text = '#fff';
                if (showAnswer) {
                  if (i === q.correctAnswer) { bg = '#166534'; border = '#22c55e'; }
                  else if (i === selectedAnswer) { bg = '#991b1b'; border = '#ef4444'; }
                  else { bg = '#1e293b'; border = '#334155'; text = '#94a3b8'; }
                } else if (selectedAnswer === i) {
                  bg = '#3b82f6'; border = '#60a5fa'; text = '#000';
                }
                return (
                  <button
                    key={i}
                    onClick={() => !showAnswer && setSelectedAnswer(i)}
                    disabled={showAnswer}
                    className="w-full p-4 rounded-xl font-semibold text-left transition-all duration-200 border-2"
                    style={{ backgroundColor: bg, borderColor: border, color: text }}
                  >
                    <span className="font-black mr-3">{String.fromCharCode(65 + i)}.</span>
                    {opt}
                  </button>
                );
              })}
            </div>
            {showAnswer && q.explanation && (
              <div className="mb-6 p-4 rounded-xl border" style={{ backgroundColor: '#0f172a', borderColor: '#334155' }}>
                <p className="text-sm text-slate-300"><span className="font-bold" style={{ color: '#60a5fa' }}>Explanation:</span> {q.explanation}</p>
              </div>
            )}
            <button
              onClick={checkAnswer}
              className="w-full font-black py-4 px-8 rounded-xl text-xl transition-all duration-200"
              style={{ backgroundColor: '#3b82f6', color: '#000' }}
            >
              {!showAnswer ? 'Check Answer' : currentQuestion + 1 === questions.length ? 'See Results' : 'Next Question'}
            </button>
          </div>
        </div>
      </div>
    );
  }

  // ── RESULTS ──
  if (gameState === 'results') {
    const pct = Math.round((score / questions.length) * 100);
    const msg = pct >= 90 ? '🏆 CNN Expert!' : pct >= 70 ? '👍 Solid understanding!' : pct >= 50 ? '👌 Getting there!' : '📚 Review the material!';
    return (
      <div className="min-h-screen bg-slate-950 text-white p-4">
        <div className="max-w-2xl mx-auto text-center">
          <div className="text-5xl mb-3">🧠</div>
          <h1 className="text-5xl font-black mb-6" style={{ color: '#60a5fa' }}>Results</h1>
          <div className="rounded-2xl p-8 border-2 mb-8" style={{ backgroundColor: '#1e293b', borderColor: '#334155' }}>
            <div className="text-6xl font-black mb-2" style={{ color: '#60a5fa' }}>{score}/{questions.length}</div>
            <div className="text-2xl font-bold mb-2">{pct}% correct</div>
            <div className="text-xl mb-8">{msg}</div>
            <div className="space-y-4 text-left mb-8">
              {questions.map((q, i) => {
                const a = answers[i];
                const ok = a && a.isCorrect;
                return (
                  <div key={i} className="p-4 rounded-xl border-2" style={{
                    borderColor: ok ? '#22c55e' : '#ef4444',
                    backgroundColor: ok ? 'rgba(22,101,52,0.2)' : 'rgba(153,27,27,0.2)',
                  }}>
                    <div className="font-semibold mb-2 text-sm">{q.question}</div>
                    <div className="text-xs">
                      <span style={{ color: ok ? '#4ade80' : '#f87171' }}>
                        Your answer: {q.options[a.selectedAnswer]}
                      </span>
                      {!ok && (
                        <div style={{ color: '#4ade80' }}>
                          Correct: {q.options[q.correctAnswer]}
                        </div>
                      )}
                      {q.explanation && (
                        <div className="mt-1 text-slate-400">{q.explanation}</div>
                      )}
                    </div>
                  </div>
                );
              })}
            </div>
            <button
              onClick={resetGame}
              className="w-full font-black py-4 px-8 rounded-xl text-xl transition-all duration-200"
              style={{ backgroundColor: '#3b82f6', color: '#000' }}
            >
              Play Again
            </button>
          </div>
        </div>
      </div>
    );
  }
};

export default TriviaGame;