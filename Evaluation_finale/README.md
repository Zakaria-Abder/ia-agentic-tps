EVALUATION DE FIN DE MODULE

Objectifs

Les systèmes RAG (Retrieval-Augmented Generation) permettent d'améliorer les réponses
des modèles de langage en s'appuyant sur des sources de connaissances externes
(Documents ou bases de donnéées). 

Les approches récentes dites Agentic RAG ajoutent des capacités de raisonnement, de 
planification, d'utilisation d'outils et de prise de décision.
Vous devez développer un agent intelligent capable de répondre à des questions complexes
à partir d'une base documentaire de votre choix. 

Le domaine est libre : Juridique , Médical , Éducation, Tourisme, Informatique , Finance , 
Recherche scientifique, etc.

L'objectif de cette évaluation est de concevoir et implémenter un système RAG agentique
complet en utilisant LangGraph au lieu d’utiliser la méthode create_agent de LangChain
comme vu en cours , car cette dernière fournit un agent prêt à l'emploi avec une boucle de
raisonnement déjà construite.

Vous devez respecter les étapes suivantes :
        • Construction de la base documentaire;
        • Développement des modèles LLM;
        • Création d'outils;
        • Développement de l’architecture du graphe en prenant en compte le state et la
        mémoire;
        • Visualisation du graphe ;
        • Évaluation du système : Vous devez tester votre système sur 10 questions simples et
        10 questions complexes, et analyser la qualité des réponses, le temps de réponse et
        la pertinence des documents récupérés.