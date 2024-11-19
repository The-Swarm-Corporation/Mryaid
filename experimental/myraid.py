from loguru import logger
import os
import random
import re
from dataclasses import dataclass
from typing import Dict, List, Optional
import json
import datetime
import numpy as np
import weaviate
from datasets import load_dataset
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from swarms import Agent
from swarm_models import OpenAIChat

load_dotenv()


@dataclass
class Persona:
    """Represents a persona from PersonaHub"""

    name: str
    description: str
    input_persona: str
    synthesized_text: str
    embedding: Optional[np.ndarray] = None

    @classmethod
    def from_dataset(cls, entry) -> "Persona":
        """Create a Persona instance from a dataset entry"""
        # Convert dataset entry to dictionary if needed
        if hasattr(entry, "keys"):
            entry = dict(entry)
        elif hasattr(entry, "_asdict"):
            entry = entry._asdict()
        elif not isinstance(entry, dict):
            # Try to access attributes directly
            try:
                synthesized_text = (
                    str(entry.synthesized_text)
                    if hasattr(entry, "synthesized_text")
                    else ""
                )
                description = (
                    str(entry.description)
                    if hasattr(entry, "description")
                    else ""
                )
                input_persona = (
                    str(entry.input_persona)
                    if hasattr(entry, "input_persona")
                    else ""
                )

                name_match = re.search(
                    r"Name:\s*([^,\n]+)", synthesized_text
                )
                name = (
                    name_match.group(1).strip()
                    if name_match
                    else "Unknown"
                )

                return cls(
                    name=name,
                    description=description,
                    input_persona=input_persona,
                    synthesized_text=synthesized_text,
                )
            except Exception as e:
                raise ValueError(
                    f"Unable to process dataset entry: {e}"
                )

        # Process dictionary entry
        synthesized_text = entry.get("synthesized_text", "")
        name_match = re.search(r"Name:\s*([^,\n]+)", synthesized_text)
        name = (
            name_match.group(1).strip() if name_match else "Unknown"
        )

        return cls(
            name=name,
            description=entry.get("description", ""),
            input_persona=entry.get("input_persona", ""),
            synthesized_text=synthesized_text,
        )


class VectorStore:
    """Weaviate integration for persona matching"""

    def __init__(self, url: str):
        self.client = weaviate.Client(
            url=os.getenv("WEAVIATE_URL"),
            auth_client_secret=weaviate.auth.AuthApiKey(
                api_key=os.getenv("WEAVIATE_API_KEY")
            ),
        )
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        self._create_schema()

    def _create_schema(self):
        """Create schema if it doesn't exist"""
        # First check if schema exists
        try:
            existing_schema = self.client.schema.get()
            if any(
                class_obj["class"] == "Persona"
                for class_obj in existing_schema["classes"]
            ):
                print("Persona schema already exists")
                return
        except Exception as e:
            print(f"Error checking schema: {e}")

        # Create schema only if it doesn't exist
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
            print("Successfully created Persona schema")
        except Exception as e:
            print(f"Error creating schema: {e}")

    def add_persona(self, persona: Persona):
        """Add persona to vector store with embedded representation"""
        text = f"{persona.description} {persona.input_persona}"
        embedding = self.embedding_model.encode(text)
        persona.embedding = embedding

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
            self.client.query.get("Persona", ["name"])
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


class SocialAgent(Agent):
    """Enhanced agent with persona-based social capabilities"""

    def __init__(self, persona: Persona, **kwargs):
        self.persona = persona
        system_prompt = self._generate_system_prompt()
        super().__init__(system_prompt=system_prompt, **kwargs)
        self.conversation_state = "initial"
        self.current_partner = None

    def _generate_system_prompt(self) -> str:
        return f"""You are {self.persona.name}. {self.persona.description}

Background: {self.persona.input_persona}

Maintain this persona while engaging in natural conversations. Follow these conversation stages:
1. Start with a friendly greeting and ask how they are
2. Share your name and ask for theirs
3. Discuss what you do, based on your persona
4. Engage in natural conversation about shared interests

Always stay in character and respond as your persona would."""

    def generate_message(self, context: str = "") -> str:
        """Generate a message based on context"""
        if not context:
            return self.run(
                "Generate a friendly greeting and ask how they are."
            )
        return self.run(
            f"Someone said: '{context}'. Respond naturally as {self.persona.name}."
        )


class ConversationManager:
    """Manages dynamic conversations between agents"""

    def __init__(self):
        self.conversation_history: Dict[str, List[Dict]] = {}

    def start_conversation(
        self,
        agent1: SocialAgent,
        agent2: SocialAgent,
        num_turns: int = 4,
    ) -> str:
        """Initialize a conversation between two agents for specified number of turns"""
        conv_id = f"conv_{len(self.conversation_history)}_{random.randint(1000, 9999)}"
        self.conversation_history[conv_id] = []

        # Initial greeting
        last_message = ""
        for _ in range(num_turns):
            # Agent 1's turn
            message1 = agent1.generate_message(last_message)
            self.add_message(conv_id, agent1.persona.name, message1)
            last_message = message1

            # Agent 2's turn
            message2 = agent2.generate_message(last_message)
            self.add_message(conv_id, agent2.persona.name, message2)
            last_message = message2

        return conv_id

    def add_message(
        self, conv_id: str, speaker_name: str, message: str
    ):
        """Add a message to the conversation history"""
        self.conversation_history[conv_id].append(
            {
                "speaker": speaker_name,
                "message": message,
                "timestamp": datetime.datetime.now().isoformat(),
            }
        )

    def export_conversations(
        self, filename: str = "conversation_history.json"
    ):
        """Export all conversations to JSON file"""
        with open(filename, "w") as f:
            json.dump(self.conversation_history, f, indent=2)


class DynamicSocialNetwork:
    """Social network with Weaviate-based matching"""

    def __init__(
        self, api_key: str, weaviate_url: str, num_agents: int = 10
    ):
        self.api_key = api_key
        self.vector_store = VectorStore(weaviate_url)
        self.persona_hub = self._load_personas(num_agents)
        self.agents: Dict[str, SocialAgent] = {}
        self.conversation_manager = ConversationManager()
        self._initialize_network()

    def _load_personas(self, num_agents: int) -> List[Persona]:
        """Load specified number of personas from PersonaHub"""
        dataset = load_dataset(
            "proj-persona/PersonaHub", "instruction"
        )
        personas = []
        for entry in dataset["train"][:num_agents]:
            try:
                persona = Persona.from_dataset(entry)
                personas.append(persona)
            except Exception as e:
                print(f"Error loading persona: {e}")
        return personas

    def _initialize_network(self):
        """Initialize the network with personas and agents"""
        for persona in self.persona_hub:
            # Add to vector store
            self.vector_store.add_persona(persona)

            # Create agent
            model = OpenAIChat(
                openai_api_key=self.api_key,
                model_name="gpt-4",
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

    def run_conversations(
        self,
        num_conversations: int = 5,
        turns_per_conversation: int = 4,
    ):
        """Run multiple conversations between compatible agents"""
        conversations = []
        available_agents = list(self.agents.keys())

        for _ in range(num_conversations):
            if len(available_agents) < 2:
                break

            # Select random initiator
            initiator_name = random.choice(available_agents)
            available_agents.remove(initiator_name)

            # Find similar partners using Weaviate
            partner_candidates = (
                self.vector_store.find_similar_personas(
                    self.agents[initiator_name].persona
                )
            )

            for partner_name in partner_candidates:
                if partner_name in available_agents:
                    available_agents.remove(partner_name)

                    # Start conversation
                    conv_id = (
                        self.conversation_manager.start_conversation(
                            self.agents[initiator_name],
                            self.agents[partner_name],
                            turns_per_conversation,
                        )
                    )

                    conversations.append(
                        (conv_id, initiator_name, partner_name)
                    )
                    break

        # Export conversation history
        self.conversation_manager.export_conversations()
        return conversations


# Example usage
def run_social_simulation(
    num_agents: int = 10,
    num_conversations: int = 5,
    turns_per_conversation: int = 4,
):
    network = DynamicSocialNetwork(
        api_key=os.getenv("OPENAI_API_KEY"),
        weaviate_url=os.getenv("WEAVIATE_URL"),
        num_agents=num_agents,
    )

    # Start multiple conversations
    conversations = network.run_conversations(
        num_conversations, turns_per_conversation
    )
    logger.info(f"Conversations: {conversations}")

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


if __name__ == "__main__":
    print(
        run_social_simulation(
            num_agents=10,
            num_conversations=5,
            turns_per_conversation=4,
        )
    )
