/**
 * Skeleton Loader with Mascot Characters
 * Adds themed loading placeholders with playful character mascots
 */

class SkeletonLoader {
  constructor() {
    this.mascots = [
      { 
        name: 'scholar', 
        path: '/static/images/mascots/scholar.svg',
        phrases: [
          "Analyzing the document structure...",
          "Extracting important concepts...",
          "Finding key information for you...",
          "Processing academic content..."
        ],
        theme: 'skeleton-theme-scholar'
      },
      { 
        name: 'robot', 
        path: '/static/images/mascots/robot.svg',
        phrases: [
          "Computing response...",
          "Processing your request...",
          "Analyzing data patterns...",
          "Running advanced algorithms..."
        ],
        theme: 'skeleton-theme-robot'
      },
      { 
        name: 'wizard', 
        path: '/static/images/mascots/wizard.svg',
        phrases: [
          "Casting knowledge spell...",
          "Conjuring information...",
          "Brewing the perfect response...",
          "Transforming content into wisdom..."
        ],
        theme: 'skeleton-theme-wizard'
      }
    ];
  }

  /**
   * Get a random mascot from available options
   * @returns {Object} Mascot configuration
   */
  getRandomMascot() {
    const randomIndex = Math.floor(Math.random() * this.mascots.length);
    return this.mascots[randomIndex];
  }

  /**
   * Get a random phrase for a given mascot
   * @param {Object} mascot - The mascot object
   * @returns {String} Random phrase
   */
  getRandomPhrase(mascot) {
    const randomIndex = Math.floor(Math.random() * mascot.phrases.length);
    return mascot.phrases[randomIndex];
  }

  /**
   * Create a document skeleton with mascot
   * @param {String} containerId - The ID of the container element
   * @param {Number} lines - Number of text lines (default: 8)
   */
  createDocumentSkeleton(containerId, lines = 8) {
    const container = document.getElementById(containerId);
    if (!container) return;
    
    const mascot = this.getRandomMascot();
    const phrase = this.getRandomPhrase(mascot);
    
    // Build HTML for document skeleton
    let html = `
      <div class="skeleton-container ${mascot.theme}">
        <img src="${mascot.path}" alt="${mascot.name} mascot" class="skeleton-mascot">
        <div class="mascot-speech">
          <span>${phrase}</span>
          <span class="typing-dots"></span>
        </div>
        <div class="skeleton-document">
          <div class="skeleton-header skeleton-loader"></div>
    `;
    
    // Add lines with varying widths
    for (let i = 0; i < lines; i++) {
      const lineClass = i % 4 === 0 ? 'short' : (i % 3 === 0 ? 'medium' : 'full');
      html += `<div class="skeleton-line ${lineClass} skeleton-loader"></div>`;
    }
    
    html += `
          <div class="loading-text">Loading document content...</div>
        </div>
      </div>
    `;
    
    container.innerHTML = html;
  }

  /**
   * Create a chat message skeleton with mascot
   * @param {String} containerId - The ID of the container element
   * @param {Boolean} isAI - Whether it's an AI message (default: true)
   */
  createChatSkeleton(containerId, isAI = true) {
    const container = document.getElementById(containerId);
    if (!container) return;
    
    const mascot = this.getRandomMascot();
    let phrase = '';
    
    if (isAI) {
      phrase = this.getRandomPhrase(mascot);
    }
    
    // Build HTML for chat message skeleton
    let html = `
      <div class="skeleton-container ${mascot.theme}">
    `;
    
    // Only add mascot for AI messages
    if (isAI) {
      html += `
        <img src="${mascot.path}" alt="${mascot.name} mascot" class="skeleton-mascot">
        <div class="mascot-speech">
          <span>${phrase}</span>
          <span class="typing-dots"></span>
        </div>
      `;
    }
    
    html += `
        <div class="skeleton-chat">
          <div class="skeleton-avatar skeleton-loader"></div>
          <div class="skeleton-content">
            <div class="skeleton-name skeleton-loader"></div>
            <div class="skeleton-text skeleton-loader"></div>
          </div>
        </div>
      </div>
    `;
    
    container.innerHTML = html;
  }

  /**
   * Create a quiz/flashcard skeleton with mascot
   * @param {String} containerId - The ID of the container element
   * @param {Number} options - Number of options (default: 4)
   */
  createCardSkeleton(containerId, options = 4) {
    const container = document.getElementById(containerId);
    if (!container) return;
    
    const mascot = this.getRandomMascot();
    const phrase = this.getRandomPhrase(mascot);
    
    // Build HTML for card skeleton
    let html = `
      <div class="skeleton-container ${mascot.theme}">
        <img src="${mascot.path}" alt="${mascot.name} mascot" class="skeleton-mascot">
        <div class="mascot-speech">
          <span>${phrase}</span>
          <span class="typing-dots"></span>
        </div>
        <div class="skeleton-card">
          <div class="skeleton-question skeleton-loader"></div>
    `;
    
    // Add option placeholders
    for (let i = 0; i < options; i++) {
      html += `<div class="skeleton-option skeleton-loader"></div>`;
    }
    
    html += `
          <div class="loading-text">Preparing quiz content...</div>
        </div>
      </div>
    `;
    
    container.innerHTML = html;
  }

  /**
   * Create multiple card skeletons
   * @param {String} containerId - The ID of the container element
   * @param {Number} count - Number of cards to create (default: 3)
   * @param {Number} options - Number of options per card (default: 4)
   */
  createMultipleCardSkeletons(containerId, count = 3, options = 4) {
    const container = document.getElementById(containerId);
    if (!container) return;
    
    container.innerHTML = '';
    
    for (let i = 0; i < count; i++) {
      const cardContainer = document.createElement('div');
      cardContainer.id = `card-skeleton-${i}`;
      container.appendChild(cardContainer);
      
      this.createCardSkeleton(cardContainer.id, options);
    }
  }
}

// Initialize and make globally available
const skeletonLoader = new SkeletonLoader();