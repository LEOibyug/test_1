// 游戏配置
const config = {
    gridSize: 20,
    speed: 150,
    initialLength: 3
};

// 游戏状态
let snake = [];
let food = null;
let direction = 'right';
let nextDirection = 'right';
let score = 0;
let gameLoop = null;

// 获取Canvas上下文
const canvas = document.getElementById('game-canvas');
const ctx = canvas.getContext('2d');
const gridWidth = canvas.width / config.gridSize;
const gridHeight = canvas.height / config.gridSize;

// 初始化游戏
function initGame() {
    // 初始化蛇
    snake = [];
    for (let i = config.initialLength - 1; i >= 0; i--) {
        snake.push({ x: i, y: 0 });
    }
    
    // 初始化方向
    direction = 'right';
    nextDirection = 'right';
    
    // 初始化分数
    score = 0;
    updateScore();
    
    // 生成第一个食物
    generateFood();
    
    // 隐藏游戏结束界面
    document.getElementById('game-over').style.display = 'none';
}

// 生成食物
function generateFood() {
    while (true) {
        food = {
            x: Math.floor(Math.random() * gridWidth),
            y: Math.floor(Math.random() * gridHeight)
        };
        
        // 确保食物不会生成在蛇身上
        if (!snake.some(segment => segment.x === food.x && segment.y === food.y)) {
            break;
        }
    }
}

// 更新分数显示
function updateScore() {
    document.getElementById('score').textContent = `分数: ${score}`;
    document.getElementById('final-score').textContent = score;
}

// 游戏主循环
function gameStep() {
    // 更新方向
    direction = nextDirection;
    
    // 获取蛇头
    const head = { ...snake[0] };
    
    // 根据方向移动蛇头
    switch (direction) {
        case 'up': head.y--; break;
        case 'down': head.y++; break;
        case 'left': head.x--; break;
        case 'right': head.x++; break;
    }
    
    // 检查碰撞
    if (isCollision(head)) {
        gameOver();
        return;
    }
    
    // 将新头部添加到蛇身数组开头
    snake.unshift(head);
    
    // 检查是否吃到食物
    if (head.x === food.x && head.y === food.y) {
        score += 10;
        updateScore();
        generateFood();
    } else {
        // 如果没有吃到食物，移除尾部
        snake.pop();
    }
    
    // 绘制游戏画面
    draw();
}

// 碰撞检测
function isCollision(head) {
    // 检查是否撞墙
    if (head.x < 0 || head.x >= gridWidth || head.y < 0 || head.y >= gridHeight) {
        return true;
    }
    
    // 检查是否撞到自己
    return snake.some(segment => segment.x === head.x && segment.y === head.y);
}

// 绘制游戏画面
function draw() {
    // 清空画布
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    // 绘制蛇
    ctx.fillStyle = '#4CAF50';
    snake.forEach((segment, index) => {
        ctx.fillRect(
            segment.x * config.gridSize,
            segment.y * config.gridSize,
            config.gridSize - 1,
            config.gridSize - 1
        );
        
        // 绘制蛇头
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
    
    // 绘制食物
    ctx.fillStyle = '#ff4444';
    ctx.fillRect(
        food.x * config.gridSize,
        food.y * config.gridSize,
        config.gridSize - 1,
        config.gridSize - 1
    );
}

// 游戏结束
function gameOver() {
    clearInterval(gameLoop);
    document.getElementById('game-over').style.display = 'block';
}

// 开始游戏
function startGame() {
    initGame();
    if (gameLoop) clearInterval(gameLoop);
    gameLoop = setInterval(gameStep, config.speed);
}

// 键盘控制
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

// 开始游戏
startGame();