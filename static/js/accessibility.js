/**
 * Accessibility features for EduVoice
 */

// Constants
const FONT_SIZES = ['small', 'medium', 'large', 'xlarge'];
const LINE_SPACINGS = ['normal', 'increased', 'double'];

// Settings object to store current accessibility preferences
let accessibilitySettings = {
    enabled: false,
    fontSize: 'medium',
    highContrast: false,
    dyslexiaFriendly: false,
    lineSpacing: 'normal',
    reduceAnimations: false,
    showReadingRuler: false
};

// Initialize accessibility features on document load
document.addEventListener('DOMContentLoaded', function() {
    initAccessibility();
    setupAccessibilityControls();
});

/**
 * Initialize accessibility features from user settings
 */
function initAccessibility() {
    // Try to get settings from localStorage if available
    const savedSettings = localStorage.getItem('accessibilitySettings');
    if (savedSettings) {
        try {
            const parsedSettings = JSON.parse(savedSettings);
            accessibilitySettings = {...accessibilitySettings, ...parsedSettings};
        } catch (e) {
            console.error('Error parsing saved accessibility settings:', e);
        }
    }
    
    // Also check if data-accessibility attributes are set on body (from server)
    const bodyEl = document.body;
    
    if (bodyEl.dataset.accessibilityEnabled === 'true') {
        accessibilitySettings.enabled = true;
    }
    
    if (bodyEl.dataset.fontSize) {
        accessibilitySettings.fontSize = bodyEl.dataset.fontSize;
    }
    
    if (bodyEl.dataset.highContrast === 'true') {
        accessibilitySettings.highContrast = true;
    }
    
    if (bodyEl.dataset.dyslexiaFriendly === 'true') {
        accessibilitySettings.dyslexiaFriendly = true;
    }
    
    if (bodyEl.dataset.lineSpacing) {
        accessibilitySettings.lineSpacing = bodyEl.dataset.lineSpacing;
    }
    
    if (bodyEl.dataset.reduceAnimations === 'true') {
        accessibilitySettings.reduceAnimations = true;
    }
    
    // Apply initial settings
    applyAccessibilitySettings();
}

/**
 * Setup event listeners for accessibility controls
 */
function setupAccessibilityControls() {
    // Toggle accessibility mode
    const accessibilityToggle = document.getElementById('accessibility-toggle');
    if (accessibilityToggle) {
        accessibilityToggle.checked = accessibilitySettings.enabled;
        accessibilityToggle.addEventListener('change', function() {
            accessibilitySettings.enabled = this.checked;
            applyAccessibilitySettings();
            saveSettings();
        });
    }
    
    // Font size controls
    const fontSizeSelect = document.getElementById('font-size-select');
    if (fontSizeSelect) {
        fontSizeSelect.value = accessibilitySettings.fontSize;
        fontSizeSelect.addEventListener('change', function() {
            accessibilitySettings.fontSize = this.value;
            applyAccessibilitySettings();
            saveSettings();
        });
    }
    
    // High contrast toggle
    const highContrastToggle = document.getElementById('high-contrast-toggle');
    if (highContrastToggle) {
        highContrastToggle.checked = accessibilitySettings.highContrast;
        highContrastToggle.addEventListener('change', function() {
            accessibilitySettings.highContrast = this.checked;
            applyAccessibilitySettings();
            saveSettings();
        });
    }
    
    // Dyslexia-friendly font toggle
    const dyslexiaFontToggle = document.getElementById('dyslexia-font-toggle');
    if (dyslexiaFontToggle) {
        dyslexiaFontToggle.checked = accessibilitySettings.dyslexiaFriendly;
        dyslexiaFontToggle.addEventListener('change', function() {
            accessibilitySettings.dyslexiaFriendly = this.checked;
            applyAccessibilitySettings();
            saveSettings();
        });
    }
    
    // Line spacing controls
    const lineSpacingSelect = document.getElementById('line-spacing-select');
    if (lineSpacingSelect) {
        lineSpacingSelect.value = accessibilitySettings.lineSpacing;
        lineSpacingSelect.addEventListener('change', function() {
            accessibilitySettings.lineSpacing = this.value;
            applyAccessibilitySettings();
            saveSettings();
        });
    }
    
    // Reduce animations toggle
    const reduceAnimationsToggle = document.getElementById('reduce-animations-toggle');
    if (reduceAnimationsToggle) {
        reduceAnimationsToggle.checked = accessibilitySettings.reduceAnimations;
        reduceAnimationsToggle.addEventListener('change', function() {
            accessibilitySettings.reduceAnimations = this.checked;
            applyAccessibilitySettings();
            saveSettings();
        });
    }
    
    // Reading ruler toggle
    const readingRulerToggle = document.getElementById('reading-ruler-toggle');
    if (readingRulerToggle) {
        readingRulerToggle.checked = accessibilitySettings.showReadingRuler;
        readingRulerToggle.addEventListener('change', function() {
            accessibilitySettings.showReadingRuler = this.checked;
            applyAccessibilitySettings();
            
            // Initialize reading ruler if enabled
            if (accessibilitySettings.showReadingRuler) {
                initReadingRuler();
            } else {
                disableReadingRuler();
            }
            
            saveSettings();
        });
    }
    
    // Save settings button - for forms that submit to server
    const saveSettingsBtn = document.getElementById('save-accessibility-settings');
    if (saveSettingsBtn) {
        saveSettingsBtn.addEventListener('click', function() {
            // Set hidden form fields with current settings
            document.getElementById('accessibility_mode').value = accessibilitySettings.enabled;
            document.getElementById('font_size').value = accessibilitySettings.fontSize;
            document.getElementById('high_contrast').value = accessibilitySettings.highContrast;
            document.getElementById('dyslexia_friendly').value = accessibilitySettings.dyslexiaFriendly;
            document.getElementById('line_spacing').value = accessibilitySettings.lineSpacing === 'normal' ? 1.5 : 
                                                           (accessibilitySettings.lineSpacing === 'increased' ? 2 : 2.5);
            document.getElementById('reduce_animations').value = accessibilitySettings.reduceAnimations;
            
            // Submit the form
            document.getElementById('accessibility-form').submit();
        });
    }
}

/**
 * Apply accessibility settings to the page
 */
function applyAccessibilitySettings() {
    const body = document.body;
    
    // Base accessibility class
    if (accessibilitySettings.enabled) {
        body.classList.add('accessibility-enabled');
    } else {
        body.classList.remove('accessibility-enabled');
        
        // If disabled, remove all other accessibility classes and return
        body.classList.remove('high-contrast', 'dyslexia-friendly', 'reduce-animations', 'show-reading-ruler');
        
        // Remove all font size classes
        FONT_SIZES.forEach(size => {
            body.classList.remove(`font-size-${size}`);
        });
        
        // Remove all line spacing classes
        LINE_SPACINGS.forEach(spacing => {
            body.classList.remove(`line-spacing-${spacing}`);
        });
        
        return;
    }
    
    // Font size
    FONT_SIZES.forEach(size => {
        body.classList.remove(`font-size-${size}`);
    });
    body.classList.add(`font-size-${accessibilitySettings.fontSize}`);
    
    // High contrast
    if (accessibilitySettings.highContrast) {
        body.classList.add('high-contrast');
    } else {
        body.classList.remove('high-contrast');
    }
    
    // Dyslexia-friendly font
    if (accessibilitySettings.dyslexiaFriendly) {
        body.classList.add('dyslexia-friendly');
    } else {
        body.classList.remove('dyslexia-friendly');
    }
    
    // Line spacing
    LINE_SPACINGS.forEach(spacing => {
        body.classList.remove(`line-spacing-${spacing}`);
    });
    body.classList.add(`line-spacing-${accessibilitySettings.lineSpacing}`);
    
    // Reduce animations
    if (accessibilitySettings.reduceAnimations) {
        body.classList.add('reduce-animations');
    } else {
        body.classList.remove('reduce-animations');
    }
    
    // Reading ruler
    if (accessibilitySettings.showReadingRuler) {
        body.classList.add('show-reading-ruler');
    } else {
        body.classList.remove('show-reading-ruler');
    }
}

/**
 * Save settings to localStorage
 */
function saveSettings() {
    try {
        localStorage.setItem('accessibilitySettings', JSON.stringify(accessibilitySettings));
    } catch (e) {
        console.error('Error saving accessibility settings:', e);
    }
}

/**
 * Initialize the reading ruler
 */
function initReadingRuler() {
    // Create reading ruler element if it doesn't exist
    let ruler = document.querySelector('.reading-ruler');
    if (!ruler) {
        ruler = document.createElement('div');
        ruler.className = 'reading-ruler';
        document.body.appendChild(ruler);
    }
    
    // Track mouse movement
    document.addEventListener('mousemove', updateReadingRulerPosition);
}

/**
 * Update the reading ruler position based on mouse movement
 */
function updateReadingRulerPosition(e) {
    const ruler = document.querySelector('.reading-ruler');
    if (ruler) {
        ruler.style.top = (e.clientY - 15) + 'px'; // Center on cursor
    }
}

/**
 * Disable the reading ruler
 */
function disableReadingRuler() {
    document.removeEventListener('mousemove', updateReadingRulerPosition);
    const ruler = document.querySelector('.reading-ruler');
    if (ruler) {
        ruler.style.display = 'none';
    }
}