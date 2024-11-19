import os
import random
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import weaviate
from datasets import load_dataset
from dotenv import load_dotenv
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer
from swarms import Agent
from swarm_models import OpenAIChat
import datetime

load_dotenv()


@dataclass
class Persona:
    """Represents a persona from PersonaHub"""

    name: str
    description: str
    input_persona: str
    synthesized_text: str

    @classmethod
    def from_dataset(cls, entry: Dict) -> "Persona":
        """Create a Persona from a dataset entry"""
        # Extract name using regex
        name_match = re.search(
            r"Name:\s*([^,\n]+)", entry["synthesized_text"]
        )
        name = (
            name_match.group(1).strip() if name_match else "Unknown"
        )

        return cls(
            name=name,
            description=entry["description"],
            input_persona=entry["input_persona"],
            synthesized_text=entry["synthesized_text"],
        )


class PersonaHub:
    """Manages loading and accessing personas"""

    def __init__(self):
        self.dataset = load_dataset(
            "proj-persona/PersonaHub", "instruction"
        )
        self.personas: List[Persona] = []
        self._load_personas()

    def _load_personas(self):
        """Load personas from the dataset"""
        for entry in self.dataset["train"]:
            try:
                persona = Persona.from_dataset(entry)
                self.personas.append(persona)
            except Exception as e:
                print(f"Error loading persona: {e}")

    def get_random_personas(self, n: int = 100) -> List[Persona]:
        """Get n random personas"""
        return random.sample(
            self.personas, min(n, len(self.personas))
        )


class SocialAgent(Agent):
    """Enhanced agent with persona-based social capabilities"""

    def __init__(self, persona: Persona, **kwargs):
        self.persona = persona
        system_prompt = self._generate_system_prompt()
        super().__init__(system_prompt=system_prompt, **kwargs)
        self.conversation_state = "initial"
        self.current_partner = None

    def _generate_system_prompt(self) -> str:
        """Generate system prompt based on persona"""
        return f"""You are {self.persona.name}. {self.persona.description}

Background: {self.persona.input_persona}

Maintain this persona while engaging in natural conversations. Follow these conversation stages:
1. Start with a friendly greeting and ask how they are
2. Share your name and ask for theirs
3. Discuss what you do, based on your persona
4. Engage in natural conversation about shared interests

Always stay in character and respond as your persona would."""

    def generate_greeting(self) -> str:
        """Generate initial greeting"""
        return self.run(
            "Generate a friendly greeting and ask how they are."
        )

    def respond_to_greeting(self, message: str) -> str:
        """Respond to a greeting and introduce yourself"""
        return self.run(
            f"Someone said: '{message}'. Respond and introduce yourself as {self.persona.name}."
        )

    def discuss_occupation(self, partner_name: str) -> str:
        """Share what you do based on your persona"""
        return self.run(
            f"Tell {partner_name} what you do, based on your persona."
        )


class ConversationManager:
    """Manages dynamic conversations between agents"""

    def __init__(self):
        self.active_conversations: Dict[str, tuple] = {}
        self.conversation_history: Dict[str, List[Dict]] = {}

    def start_conversation(
        self, agent1: SocialAgent, agent2: SocialAgent
    ) -> str:
        """Initialize a natural conversation between two agents"""
        conv_id = f"conv_{len(self.conversation_history)}_{random.randint(1000, 9999)}"
        self.active_conversations[conv_id] = (agent1, agent2)
        self.conversation_history[conv_id] = []

        # Initial greeting
        greeting = agent1.generate_greeting()
        self.add_message(conv_id, agent1.persona.name, greeting)

        # Response and introduction
        response = agent2.respond_to_greeting(greeting)
        self.add_message(conv_id, agent2.persona.name, response)

        # Share occupations
        occupation_share = agent1.discuss_occupation(
            agent2.persona.name
        )
        self.add_message(
            conv_id, agent1.persona.name, occupation_share
        )

        return conv_id

    def add_message(
        self, conv_id: str, speaker_name: str, message: str
    ):
        """Add a message to the conversation history"""
        self.conversation_history[conv_id].append(
            {
                "speaker": speaker_name,
                "message": message,
                "timestamp": datetime.now().isoformat(),
            }
        )


@dataclass
class Persona:
    """Represents a persona from PersonaHub"""

    name: str
    description: str
    input_persona: str
    synthesized_text: str
    embedding: Optional[np.ndarray] = None

    @classmethod
    def from_dataset(cls, entry: Dict) -> "Persona":
        name_match = re.search(
            r"Name:\s*([^,\n]+)", entry["synthesized_text"]
        )
        name = (
            name_match.group(1).strip() if name_match else "Unknown"
        )
        return cls(
            name=name,
            description=entry["description"],
            input_persona=entry["input_persona"],
            synthesized_text=entry["synthesized_text"],
        )


class VectorStore:
    """Enhanced Weaviate integration for persona matching"""

    def __init__(self, url: str):
        self.client = weaviate.Client(url)
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        self._create_schema()

    def _create_schema(self):
        schema = {
            "classes": [
                {
                    "class": "Persona",
                    "vectorizer": "none",
                    "properties": [
                        {"name": "name", "dataType": ["string"]},
                        {"name": "description", "dataType": ["text"]},
                        {
                            "name": "inputPersona",
                            "dataType": ["text"],
                        },
                        {
                            "name": "interests",
                            "dataType": ["string[]"],
                        },
                        {
                            "name": "availability",
                            "dataType": ["boolean"],
                        },
                    ],
                }
            ]
        }
        try:
            self.client.schema.create(schema)
        except Exception as e:
            print(f"Schema already exists: {e}")

    def add_persona(self, persona: Persona):
        """Add persona to vector store with embedded representation"""
        # Generate embedding from combined persona information
        text = f"{persona.description} {persona.input_persona}"
        embedding = self.embedding_model.encode(text)
        persona.embedding = embedding

        # Extract interests using NLP (simplified version)
        interests = re.findall(
            r"interested in|likes|enjoys|passionate about\s+([^,.]+)",
            persona.synthesized_text.lower(),
        )

        self.client.data_object.create(
            class_name="Persona",
            data_object={
                "name": persona.name,
                "description": persona.description,
                "inputPersona": persona.input_persona,
                "interests": interests if interests else [],
                "availability": True,
            },
            vector=embedding.tolist(),
        )

    def find_similar_personas(
        self, persona: Persona, limit: int = 5
    ) -> List[str]:
        """Find similar personas using vector similarity"""
        result = (
            self.client.query.get("Persona", ["name", "interests"])
            .with_near_vector({"vector": persona.embedding.tolist()})
            .with_where(
                {
                    "path": ["availability"],
                    "operator": "Equal",
                    "valueBoolean": True,
                }
            )
            .with_limit(limit + 1)
            .do()
        )

        matches = result["data"]["Get"]["Persona"]
        return [
            match["name"]
            for match in matches
            if match["name"] != persona.name
        ]


class GraphStore:
    """Enhanced Neo4j integration for social relationships"""

    def __init__(self, uri: str, user: str, password: str):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self._initialize_db()

    def _initialize_db(self):
        with self.driver.session() as session:
            # Create constraints
            session.run(
                "CREATE CONSTRAINT persona_name IF NOT EXISTS "
                "FOR (p:Persona) REQUIRE p.name IS UNIQUE"
            )

            # Create indexes
            session.run(
                "CREATE INDEX persona_interests IF NOT EXISTS "
                "FOR (p:Persona) ON (p.interests)"
            )

    def add_persona(self, persona: Persona):
        """Add persona to graph database with relationships"""
        with self.driver.session() as session:
            # Create persona node with interests
            interests = re.findall(
                r"interested in|likes|enjoys|passionate about\s+([^,.]+)",
                persona.synthesized_text.lower(),
            )

            session.run(
                """
                MERGE (p:Persona {name: $name})
                SET p.description = $description,
                    p.interests = $interests,
                    p.availability = true
                """,
                name=persona.name,
                description=persona.description,
                interests=interests,
            )

            # Create interest relationships
            for interest in interests:
                session.run(
                    """
                    MERGE (i:Interest {name: $interest})
                    WITH i
                    MATCH (p:Persona {name: $name})
                    MERGE (p)-[:INTERESTED_IN]->(i)
                    """,
                    interest=interest.strip(),
                    name=persona.name,
                )

    def record_interaction(
        self,
        persona1: str,
        persona2: str,
        interaction_type: str = "CONVERSED_WITH",
    ):
        """Record interaction between personas"""
        with self.driver.session() as session:
            session.run(
                """
                MATCH (p1:Persona {name: $name1})
                MATCH (p2:Persona {name: $name2})
                MERGE (p1)-[r:INTERACTION]->(p2)
                ON CREATE SET r.count = 1, r.type = $type
                ON MATCH SET r.count = r.count + 1,
                            r.lastInteraction = datetime()
                """,
                name1=persona1,
                name2=persona2,
                type=interaction_type,
            )

    def find_recommended_partners(
        self, persona_name: str, limit: int = 5
    ) -> List[str]:
        """Find recommended conversation partners based on graph patterns"""
        with self.driver.session() as session:
            result = session.run(
                """
                MATCH (p:Persona {name: $name})-[:INTERESTED_IN]->(i:Interest)
                MATCH (other:Persona)-[:INTERESTED_IN]->(i)
                WHERE other.name <> $name
                AND other.availability = true
                WITH other, count(distinct i) as shared_interests,
                     collect(distinct i.name) as interests
                ORDER BY shared_interests DESC, rand()
                LIMIT $limit
                RETURN other.name as name, shared_interests, interests
                """,
                name=persona_name,
                limit=limit,
            )

            return [
                (
                    record["name"],
                    record["shared_interests"],
                    record["interests"],
                )
                for record in result
            ]


class DynamicSocialNetwork:
    """Integrated social network with enhanced matching capabilities"""

    def __init__(
        self,
        api_key: str,
        weaviate_url: str,
        neo4j_uri: str,
        neo4j_user: str,
        neo4j_password: str,
    ):
        self.api_key = api_key
        self.vector_store = VectorStore(weaviate_url)
        self.graph_store = GraphStore(
            neo4j_uri, neo4j_user, neo4j_password
        )
        self.persona_hub = self._load_personas()
        self.agents: Dict[str, SocialAgent] = {}
        self.conversation_manager = ConversationManager()

        self._initialize_network()

    def _load_personas(self) -> List[Persona]:
        """Load personas from PersonaHub"""
        dataset = load_dataset(
            "proj-persona/PersonaHub", "instruction"
        )
        personas = []
        for entry in dataset["train"][
            :100
        ]:  # Load first 100 personas
            try:
                persona = Persona.from_dataset(entry)
                personas.append(persona)
            except Exception as e:
                print(f"Error loading persona: {e}")
        return personas

    def _initialize_network(self):
        """Initialize the network with personas and agents"""
        for persona in self.persona_hub:
            # Add to vector and graph stores
            self.vector_store.add_persona(persona)
            self.graph_store.add_persona(persona)

            # Create agent
            model = OpenAIChat(
                openai_api_key=self.api_key,
                model_name="gpt-4-mini",
                temperature=0.7,
            )

            agent = SocialAgent(
                persona=persona,
                agent_name=f"Agent-{persona.name}",
                llm=model,
                max_loops=1,
                autosave=True,
                verbose=True,
            )

            self.agents[persona.name] = agent

    def find_conversation_partners(
        self, persona_name: str, n: int = 3
    ) -> List[Tuple[str, float, List[str]]]:
        """Find potential conversation partners using both vector and graph approaches"""
        # Get vector-based matches
        vector_matches = set(
            self.vector_store.find_similar_personas(
                self.agents[persona_name].persona, limit=n
            )
        )

        # Get graph-based recommendations
        graph_matches = self.graph_store.find_recommended_partners(
            persona_name, limit=n
        )
        graph_names = {match[0] for match in graph_matches}

        # Combine and score matches
        all_matches = []
        for name in vector_matches.union(graph_names):
            score = (
                1
                if name in vector_matches
                else 0 + 2 if name in graph_names else 0
            )
            interests = next(
                (m[2] for m in graph_matches if m[0] == name), []
            )
            all_matches.append((name, score, interests))

        # Sort by score and return top N
        all_matches.sort(key=lambda x: x[1], reverse=True)
        return all_matches[:n]

    def initiate_conversations(self, num_conversations: int = 5):
        """Start multiple dynamic conversations between compatible agents"""
        available_agents = list(self.agents.keys())
        conversations = []

        for _ in range(num_conversations):
            if len(available_agents) < 2:
                break

            # Select random initiator
            initiator_name = random.choice(available_agents)
            available_agents.remove(initiator_name)

            # Find compatible partners
            partners = self.find_conversation_partners(initiator_name)
            if partners:
                partner_name = partners[0][0]
                if partner_name in available_agents:
                    available_agents.remove(partner_name)

                    # Start conversation
                    conv_id = (
                        self.conversation_manager.start_conversation(
                            self.agents[initiator_name],
                            self.agents[partner_name],
                        )
                    )

                    # Record interaction
                    self.graph_store.record_interaction(
                        initiator_name, partner_name
                    )

                    conversations.append(
                        (conv_id, initiator_name, partner_name)
                    )

        return conversations


# Example usage
network = DynamicSocialNetwork(
    api_key=os.getenv("OPENAI_API_KEY"),
    weaviate_url=os.getenv("WEAVIATE_URL"),
    neo4j_uri=os.getenv("NEO4J_URI"),
    neo4j_user=os.getenv("NEO4J_USER"),
    neo4j_password=os.getenv("NEO4J_PASSWORD"),
)

# Start multiple conversations
conversations = network.initiate_conversations(5)

# Print conversation details
for conv_id, initiator, partner in conversations:
    print(f"\nConversation {conv_id}:")
    print(f"Between {initiator} and {partner}")
    for (
        message
    ) in network.conversation_manager.conversation_history[
        conv_id
    ]:
        print(f"{message['speaker']}: {message['message']}")
    print(
        "\nShared Interests:",
        network.find_conversation_partners(initiator)[0][2],
    )