/**
 * Floating Action Button functionality
 */
document.addEventListener('DOMContentLoaded', function() {
    const fab = document.querySelector('.floating-action-btn');
    const fabMenu = document.querySelector('.fab-menu');
    
    // Toggle the FAB menu when the main button is clicked
    if (fab && fabMenu) {
        fab.addEventListener('click', function() {
            fabMenu.classList.toggle('show');
            
            // Change the icon from plus to x when opened
            const icon = fab.querySelector('i');
            if (icon) {
                if (fabMenu.classList.contains('show')) {
                    icon.classList.remove('bi-plus-lg');
                    icon.classList.add('bi-x-lg');
                } else {
                    icon.classList.remove('bi-x-lg');
                    icon.classList.add('bi-plus-lg');
                }
            }
        });
        
        // Close FAB menu when clicking outside
        document.addEventListener('click', function(event) {
            if (!fab.contains(event.target) && !fabMenu.contains(event.target) && fabMenu.classList.contains('show')) {
                fabMenu.classList.remove('show');
                const icon = fab.querySelector('i');
                if (icon) {
                    icon.classList.remove('bi-x-lg');
                    icon.classList.add('bi-plus-lg');
                }
            }
        });
    }
});