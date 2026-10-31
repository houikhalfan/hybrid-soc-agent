// ============================================================
// NŒUD CODE : "Vérif Disponibilité Analyste"
// À placer JUSTE APRÈS "Response Agent"
// ============================================================
// Heuristique simple : un analyste est considéré disponible pendant
// les heures ouvrées (9h-19h, en semaine). En dehors -> confinement
// automatique si l'action le nécessite. À affiner plus tard avec un
// vrai planning d'astreinte si besoin.

const item = $json;

const now = new Date();
const hour = now.getUTCHours();
const day = now.getUTCDay();

const isWorkingHours = hour >= 9 && hour < 19;
const isWeekday = day >= 1 && day <= 5;

item.analyst_available = isWorkingHours && isWeekday;

return item;
