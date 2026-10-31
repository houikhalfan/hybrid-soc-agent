// ============================================================
// NŒUD CODE : "Préparation Corpus RAG"
// À placer AVANT "Loop Over Items" (une seule fois, pas par incident)
// ============================================================
// Ce nœud produit la liste de documents à insérer dans la base
// vectorielle -- chaque document = une technique MITRE ATT&CK avec
// sa description, ses signes typiques, et son playbook de réponse.
// C'est ce corpus que l'Agent Connaissance interrogera par similarité.

const documents = [
  {
    pageContent: "Technique: T1046 - Network Service Discovery (Scan de ports). Attaque: PortScan. Indicateurs typiques: grand nombre de ports de destination distincts contactés par une même source vers une même destination en peu de temps, faible volume par flux, faible durée par connexion. Règles de détection: port_scan_triggered = True, n_distinct_dst_ports élevé. Sévérité: Moyenne (phase de reconnaissance, précède souvent une attaque ciblée). Playbook SOC: 1. Surveiller la source. 2. Vérifier si le scan correspond à un audit interne autorisé. 3. Ne pas bloquer immédiatement sauf récurrence ou combinaison avec d'autres signaux. Notes faux positifs: des outils de supervision réseau légitimes (monitoring, inventaire) peuvent contacter de nombreux ports périodiquement.",
    metadata: { technique: "T1046", type_attaque: "PortScan" }
  },
  {
    pageContent: "Technique: T1498 - Network Denial of Service (DoS/DDoS volumétrique). Attaques: DDoS, DoS Hulk, DoS GoldenEye. Indicateurs typiques: très grand nombre de flux en peu de temps vers une même destination, un seul ou très peu de ports distincts contactés, volume de données élevé, forte proportion de flux jugés anormaux par Isolation Forest. Règles de détection: large_flow_triggered = True, n_if_anomalies élevé, n_distinct_dst_ports faible. Sévérité: Élevée (indisponibilité de service). Playbook SOC: 1. Vérifier le volume de trafic. 2. Vérifier la saturation CPU/bande passante de la cible. 3. Activer la limitation de débit (rate limiting). 4. Bloquer la source si le volume est confirmé anormal. Notes faux positifs: du trafic de sauvegarde planifié (backup) peut ressembler à ce comportement par son volume élevé mais reste généralement récurrent et prévisible.",
    metadata: { technique: "T1498", type_attaque: "DDoS/DoS Hulk/DoS GoldenEye" }
  },
  {
    pageContent: "Technique: T1499 - Endpoint Denial of Service (DoS applicatif lent, épuisement de ressources). Attaques: DoS slowloris, DoS Slowhttptest. Indicateurs typiques: connexions de durée anormalement longue, faible volume de données par connexion, nombreuses connexions maintenues ouvertes simultanément. Règles de détection: long_connection_triggered = True, volume faible malgré durée élevée. Sévérité: Élevée (indisponibilité progressive du service). Playbook SOC: 1. Vérifier la configuration des timeouts serveur. 2. Compter les connexions simultanées ouvertes par la source. 3. Bloquer temporairement la source si le pattern persiste. Notes faux positifs: des connexions longues légitimes existent pour certains protocoles de streaming ou de maintenance à distance.",
    metadata: { technique: "T1499", type_attaque: "DoS slowloris/DoS Slowhttptest" }
  },
  {
    pageContent: "Technique: T1110 - Brute Force. Attaques: SSH-Patator, FTP-Patator. Indicateurs typiques: nombreuses tentatives de connexion rapprochées vers le même port sensible (22, 21, 3389...), échecs répétés suivis parfois d'un succès. Règles de détection: brute_force_triggered = True, comptage élevé de tentatives sur un port sensible dans une fenêtre courte. Sévérité: Élevée (risque de compromission d'accès). Playbook SOC: 1. Bloquer temporairement la source. 2. Vérifier les logs d'authentification pour une connexion réussie après les échecs. 3. Forcer une rotation des mots de passe si compromission suspectée. Notes faux positifs: un utilisateur légitime ayant oublié son mot de passe peut générer plusieurs échecs, mais rarement autant qu'un script automatisé.",
    metadata: { technique: "T1110", type_attaque: "SSH-Patator/FTP-Patator" }
  },
  {
    pageContent: "Technique: T1071 - Application Layer Protocol (communication de commande et contrôle, C2). Attaque: Bot. Indicateurs typiques: communications périodiques et régulières entre une machine interne et une destination externe, souvent peu de flux mais réguliers dans le temps, utilisant des protocoles standards (HTTP, DNS) pour se fondre dans le trafic normal. Règles de détection: souvent PEU de règles classiques se déclenchent (comportement discret par nature), score de déviation baseline peut être notable. Sévérité: Critique (machine potentiellement déjà compromise). Playbook SOC: 1. Isoler immédiatement la machine du réseau. 2. Analyser la machine pour rechercher un malware. 3. Identifier le serveur de destination comme indicateur de compromission (IOC). Notes faux positifs: certains services légitimes (mises à jour automatiques, télémétrie) ont un pattern de communication périodique similaire.",
    metadata: { technique: "T1071", type_attaque: "Bot" }
  },
  {
    pageContent: "Technique: T1110.001 - Password Guessing sur application web. Attaque: Web Attack Brute Force. Indicateurs typiques: similaire au brute force réseau mais ciblant un formulaire d'authentification web (port 80/443) plutôt qu'un protocole comme SSH. Sévérité: Élevée. Playbook SOC: 1. Bloquer temporairement la source. 2. Activer un CAPTCHA ou une authentification à deux facteurs si disponible. Notes faux positifs: rare en dehors d'un test de charge légitime sur l'application.",
    metadata: { technique: "T1110.001", type_attaque: "Web Attack Brute Force" }
  },
  {
    pageContent: "Technique: T1190 - Exploit Public-Facing Application. Attaques: Infiltration, Heartbleed. Indicateurs typiques: exploitation d'une vulnérabilité connue d'une application exposée, peut ne générer que très peu de flux mais avec un impact potentiellement sévère. Sévérité: Critique. Playbook SOC: 1. Isoler la machine immédiatement. 2. Escalader vers un analyste senior. 3. Patcher en urgence le service vulnérable identifié. 4. Régénérer les certificats et clés si une fuite de mémoire est suspectée (cas Heartbleed). Notes faux positifs: très rares compte tenu de la spécificité de la signature d'exploitation.",
    metadata: { technique: "T1190", type_attaque: "Infiltration/Heartbleed" }
  },
  {
    pageContent: "Trafic bénin (BENIGN). Aucune technique d'attaque connue ne correspond. Indicateurs typiques: le comportement observé correspond au trafic réseau normal attendu pour ce type de machine, même si certains seuils de règles automatiques se sont déclenchés par excès de prudence. Sévérité: Aucune. Playbook SOC: 1. Aucune action requise. 2. Journaliser pour référence. 3. Surveiller si le pattern devient récurrent ou s'intensifie. Notes faux positifs: les serveurs très actifs (DNS, passerelles, serveurs de fichiers) génèrent souvent ce type de faux positifs sur les règles de scan de ports ou de volume.",
    metadata: { technique: "N/A", type_attaque: "BENIGN" }
  }
];

return documents.map(doc => ({ json: doc }));
