// ��Ϸ����
const config = {
    gridSize: 20,
    speed: 150,
    initialLength: 3
};

// ��Ϸ״̬
let snake = [];
let food = null;
let direction = 'right';
let nextDirection = 'right';
let score = 0;
let gameLoop = null;

// ��ȡCanvas������
const canvas = document.getElementById('game-canvas');
const ctx = canvas.getContext('2d');
const gridWidth = canvas.width / config.gridSize;
const gridHeight = canvas.height / config.gridSize;

// ��ʼ����Ϸ
function initGame() {
    // ��ʼ����
    snake = [];
    for (let i = config.initialLength - 1; i >= 0; i--) {
        snake.push({ x: i, y: 0 });
    }
    
    // ��ʼ������
    direction = 'right';
    nextDirection = 'right';
    
    // ��ʼ������
    score = 0;
    updateScore();
    
    // ���ɵ�һ��ʳ��
    generateFood();
    
    // ������Ϸ��������
    document.getElementById('game-over').style.display = 'none';
}

// ����ʳ��
function generateFood() {
    while (true) {
        food = {
            x: Math.floor(Math.random() * gridWidth),
            y: Math.floor(Math.random() * gridHeight)
        };
        
        // ȷ��ʳ�ﲻ��������������
        if (!snake.some(segment => segment.x === food.x && segment.y === food.y)) {
            break;
        }
    }
}

// ���·�����ʾ
function updateScore() {
    document.getElementById('score').textContent = `����: ${score}`;
    document.getElementById('final-score').textContent = score;
}

// ��Ϸ��ѭ��
function gameStep() {
    // ���·���
    direction = nextDirection;
    
    // ��ȡ��ͷ
    const head = { ...snake[0] };
    
    // ���ݷ����ƶ���ͷ
    switch (direction) {
        case 'up': head.y--; break;
        case 'down': head.y++; break;
        case 'left': head.x--; break;
        case 'right': head.x++; break;
    }
    
    // �����ײ
    if (isCollision(head)) {
        gameOver();
        return;
    }
    
    // ����ͷ����ӵ��������鿪ͷ
    snake.unshift(head);
    
    // ����Ƿ�Ե�ʳ��
    if (head.x === food.x && head.y === food.y) {
        score += 10;
        updateScore();
        generateFood();
    } else {
        // ���û�гԵ�ʳ��Ƴ�β��
        snake.pop();
    }
    
    // ������Ϸ����
    draw();
}

// ��ײ���
function isCollision(head) {
    // ����Ƿ�ײǽ
    if (head.x < 0 || head.x >= gridWidth || head.y < 0 || head.y >= gridHeight) {
        return true;
    }
    
    // ����Ƿ�ײ���Լ�
    return snake.some(segment => segment.x === head.x && segment.y === head.y);
}

// ������Ϸ����
function draw() {
    // ��ջ���
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    // ������
    ctx.fillStyle = '#4CAF50';
    snake.forEach((segment, index) => {
        ctx.fillRect(
            segment.x * config.gridSize,
            segment.y * config.gridSize,
            config.gridSize - 1,
            config.gridSize - 1
        );
        
        // ������ͷ
        if (index === 0) {
            ctx.fillStyle = '#45a049';
            ctx.fillRect(
                segment.x * config.gridSize,
                segment.y * config.gridSize,
                config.gridSize - 1,
                config.gridSize - 1
            );
        }
    });
    
    // ����ʳ��
    ctx.fillStyle = '#ff4444';
    ctx.fillRect(
        food.x * config.gridSize,
        food.y * config.gridSize,
        config.gridSize - 1,
        config.gridSize - 1
    );
}

// ��Ϸ����
function gameOver() {
    clearInterval(gameLoop);
    document.getElementById('game-over').style.display = 'block';
}

// ��ʼ��Ϸ
function startGame() {
    initGame();
    if (gameLoop) clearInterval(gameLoop);
    gameLoop = setInterval(gameStep, config.speed);
}

// ���̿���
document.addEventListener('keydown', (event) => {
    switch (event.key) {
        case 'ArrowUp':
            if (direction !== 'down') nextDirection = 'up';
            break;
        case 'ArrowDown':
            if (direction !== 'up') nextDirection = 'down';
            break;
        case 'ArrowLeft':
            if (direction !== 'right') nextDirection = 'left';
            break;
        case 'ArrowRight':
            if (direction !== 'left') nextDirection = 'right';
            break;
    }
});

// ��ʼ��Ϸ
startGame();