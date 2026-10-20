document.addEventListener('DOMContentLoaded', () => {
  const tabs = document.querySelectorAll('[data-result-tab]');
  const panels = document.querySelectorAll('.result-panel');

  tabs.forEach((tab) => {
    tab.addEventListener('click', () => {
      const targetId = tab.dataset.resultTab;

      tabs.forEach((item) => {
        const isActive = item === tab;
        item.classList.toggle('is-link', isActive);
        item.setAttribute('aria-selected', String(isActive));
      });

      panels.forEach((panel) => {
        const isActive = panel.id === targetId;
        panel.classList.toggle('is-hidden', !isActive);
        panel.hidden = !isActive;
      });
    });
  });
});
