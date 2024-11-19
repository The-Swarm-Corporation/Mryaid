from loguru import logger
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import json
import datetime
import numpy as np
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from swarms import Agent
from swarm_models import OpenAIChat



# Configure detailed logging
logger.remove()  # Remove default handler
logger.add(
    "social_simulation.log",
    rotation="500 MB",
    level="DEBUG",  # Changed to DEBUG for more detailed logging
    format="{time:YYYY-MM-DD at HH:mm:ss} | {level} | {function}:{line} | {message}",
    backtrace=True,  # Enable backtrace
    diagnose=True,  # Enable diagnosis
)

# Add console logging for immediate feedback
logger.add(
    lambda msg: print(msg),
    level="DEBUG",
    format="{level} | {message}",
)



@dataclass
class Persona:
    """Represents a persona from PersonaHub with friends"""

    name: str
    description: str
    input_persona: str
    synthesized_text: str
    friends: List[str]
    personality: Dict[str, str]  # Added to store personality traits
    embedding: Optional[np.ndarray] = None

    @classmethod
    def from_dataset(cls, entry) -> "Persona":
        """Create a Persona instance from a dataset entry"""
        try:
            logger.debug(f"Raw entry: {entry}")

            # Direct field access from dataset
            name = entry.get("name", "Unknown")
            description = entry.get("description", "")
            input_persona = entry.get("input_persona", "")
            synthesized_text = entry.get("synthesized_text", "")

            # Extract friends array
            friends_data = entry.get("friends", [])
            if isinstance(friends_data, str):
                # If friends is stored as a string, convert to list
                friends = [
                    friend.strip()
                    for friend in friends_data.split(",")
                ]
            else:
                friends = list(friends_data) if friends_data else []

            # Extract personality traits if available
            personality = entry.get("personality", {})
            if isinstance(personality, str):
                # If personality is stored as a string, convert to dict
                try:
                    personality = json.loads(personality)
                except json.JSONDecodeError:
                    personality = {"traits": personality}

            logger.debug(
                f"""
            Creating persona with:
            - Name: {name}
            - Description length: {len(description)}
            - Input persona length: {len(input_persona)}
            - Synthesized text length: {len(synthesized_text)}
            - Number of friends: {len(friends)}
            - Personality traits: {personality}
            """
            )

            return cls(
                name=name,
                description=description,
                input_persona=input_persona,
                synthesized_text=synthesized_text,
                friends=friends,
                personality=personality,
            )

        except Exception as e:
            logger.error(f"Error creating persona: {e}")
            logger.error(f"Entry that caused error: {entry}")
            raise ValueError(
                f"Failed to create persona from entry: {e}"
            )


class LocalVectorStore:
    """Local vector store for persona matching using numpy"""

    def __init__(self):
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        self.personas: Dict[str, Persona] = {}
        self.embeddings: Dict[str, np.ndarray] = {}
        logger.info(
            "Initialized LocalVectorStore with embedding model"
        )

    def add_persona(self, persona: Persona) -> None:
        """Add persona to vector store with embedded representation"""
        try:
            text = f"{persona.description} {persona.input_persona}"
            embedding = self.embedding_model.encode(text)
            persona.embedding = embedding

            self.personas[persona.name] = persona
            self.embeddings[persona.name] = embedding

            logger.info(
                f"Added persona {persona.name} to vector store"
            )
        except Exception as e:
            logger.error(
                f"Error adding persona {persona.name} to vector store: {e}"
            )
            raise

    def cosine_similarity(
        self, a: np.ndarray, b: np.ndarray
    ) -> float:
        """Calculate cosine similarity between two vectors"""
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    def find_similar_personas(
        self, persona: Persona, limit: int = 5
    ) -> List[Tuple[str, float]]:
        """Find similar personas using cosine similarity"""
        try:
            if persona.name not in self.embeddings:
                logger.warning(
                    f"Persona {persona.name} not found in vector store"
                )
                return []

            query_embedding = self.embeddings[persona.name]
            similarities = []

            for name, embedding in self.embeddings.items():
                if name != persona.name:
                    similarity = self.cosine_similarity(
                        query_embedding, embedding
                    )
                    similarities.append((name, similarity))

            # Sort by similarity score in descending order
            similarities.sort(key=lambda x: x[1], reverse=True)

            logger.info(
                f"Found {len(similarities[:limit])} similar personas for {persona.name}"
            )
            return similarities[:limit]

        except Exception as e:
            logger.error(f"Error finding similar personas: {e}")
            return []


class DynamicSocialNetwork:
    """Social network with local vector-based matching"""

    def __init__(self, api_key: str, num_agents: int = 10):
        self.api_key = api_key
        self.vector_store = LocalVectorStore()
        self.persona_hub = self._load_personas(num_agents)
        self.agents: Dict[str, SocialAgent] = {}
        self.conversation_manager = ConversationManager()
        self._initialize_network()
        logger.info(
            f"Initialized DynamicSocialNetwork with {num_agents} agents"
        )

    def _load_personas(self, num_agents: int) -> List[Persona]:
        """Load specified number of personas from PersonaHub"""
        try:
            # Load the dataset
            dataset = load_dataset("proj-persona/PersonaHub", "npc")
            personas = []

            # Debug print the dataset structure
            logger.debug("Dataset Structure:")
            logger.debug(f"Available splits: {dataset.keys()}")
            logger.debug(
                f"First entry structure: {dataset['train'][0]}"
            )
            logger.debug(
                f"First entry keys: {dataset['train'][0].keys()}"
            )

            # Process entries with proper field access
            for i, entry in enumerate(dataset["train"][:num_agents]):
                try:
                    if not isinstance(entry, dict):
                        entry = dict(entry)

                    logger.debug(f"Processing entry {i}:")
                    logger.debug(f"Entry type: {type(entry)}")
                    logger.debug(f"Entry content: {entry}")

                    persona = Persona.from_dataset(entry)
                    personas.append(persona)
                    logger.info(
                        f"Successfully loaded persona: {persona.name}"
                    )

                except Exception as e:
                    logger.error(
                        f"Error loading individual persona {i}: {e}"
                    )
                    logger.error(f"Entry that caused error: {entry}")
                    continue

            if not personas:
                logger.warning(
                    "No personas were successfully loaded!"
                )

            logger.info(
                f"Successfully loaded {len(personas)} personas"
            )
            return personas

        except Exception as e:
            logger.error(f"Error loading personas from dataset: {e}")
            raise

    def _initialize_network(self):
        """Initialize the network with personas and agents"""
        try:
            for persona in self.persona_hub:
                # Add to vector store
                self.vector_store.add_persona(persona)

                # Create agent
                model = OpenAIChat(
                    openai_api_key=self.api_key,
                    model_name="gpt-4o-mini",
                    temperature=1.2,
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

            logger.info(
                f"Initialized network with {len(self.agents)} agents"
            )

        except Exception as e:
            logger.error(f"Error initializing network: {e}")
            raise

    def run_conversations(
        self,
        num_conversations: int = 5,
        turns_per_conversation: int = 4,
    ) -> List[Tuple[str, str, str]]:
        """Run multiple conversations between compatible agents"""
        try:
            conversations = []
            available_agents = list(self.agents.keys())

            for i in range(num_conversations):
                if len(available_agents) < 2:
                    logger.warning(
                        "Not enough available agents for more conversations"
                    )
                    break

                # Select random initiator
                initiator_name = random.choice(available_agents)
                available_agents.remove(initiator_name)

                # Find similar partners using local vector store
                partner_candidates = (
                    self.vector_store.find_similar_personas(
                        self.agents[initiator_name].persona
                    )
                )

                for (
                    partner_name,
                    similarity_score,
                ) in partner_candidates:
                    if partner_name in available_agents:
                        available_agents.remove(partner_name)
                        logger.info(
                            f"Matching {initiator_name} with {partner_name} (similarity: {similarity_score:.3f})"
                        )

                        # Start conversation
                        conv_id = self.conversation_manager.start_conversation(
                            self.agents[initiator_name],
                            self.agents[partner_name],
                            turns_per_conversation,
                        )

                        conversations.append(
                            (conv_id, initiator_name, partner_name)
                        )
                        break

            # Export conversation history
            self.conversation_manager.export_conversations()
            logger.info(
                f"Completed {len(conversations)} conversations"
            )
            return conversations

        except Exception as e:
            logger.error(f"Error running conversations: {e}")
            return []


class SocialAgent(Agent):
    """Enhanced agent with persona-based social capabilities and awareness of friends"""

    def __init__(self, persona: Persona, **kwargs):
        self.persona = persona
        system_prompt = self._generate_system_prompt()
        super().__init__(system_prompt=system_prompt, **kwargs)
        self.conversation_state = "initial"
        self.current_partner = None
        logger.info(
            f"Initialized SocialAgent for persona: {persona.name}"
        )

    def _generate_system_prompt(self) -> str:
        # Format friends list for prompt
        friends_str = (
            ", ".join(self.persona.friends)
            if self.persona.friends
            else "no close friends yet"
        )

        system_prompt = f"""You are {self.persona.name}.

Background Information:
- Description: {self.persona.description}
- Personal History: {self.persona.input_persona}
- Friends: {friends_str}

Additional Context from Generated Profile:
{self.persona.synthesized_text}

Conversation Guidelines:
1. Always stay in character as {self.persona.name}
2. Reference your background and experiences naturally
3. If talking to one of your friends ({friends_str}), show familiarity
4. Share appropriate personal details from your background
5. Express interest in learning about others
6. Maintain consistent personality traits

Remember your relationships and adapt your tone accordingly."""

        logger.debug(
            f"Generated system prompt for {self.persona.name}"
        )
        return system_prompt

    def generate_message(
        self, context: str = "", partner_name: str = None
    ) -> str:
        """Generate a message based on context and conversation partner"""
        try:
            is_friend = (
                partner_name in self.persona.friends
                if partner_name
                else False
            )

            if not context:
                if is_friend:
                    prompt = f"Generate a warm, familiar greeting for your friend {partner_name}."
                else:
                    prompt = f"Generate a friendly greeting for {partner_name}, who you're meeting for the first time."
            else:
                relationship_context = (
                    "your friend"
                    if is_friend
                    else "someone new named"
                )
                prompt = f"{relationship_context} {partner_name} said: '{context}'. Respond naturally as {self.persona.name}."

            message = self.run(prompt)
            logger.info(
                f"Generated message from {self.persona.name} to {partner_name}: {message[:50]}..."
            )
            return message

        except Exception as e:
            logger.error(
                f"Error generating message for {self.persona.name}: {e}"
            )
            return f"Hello! [Error: {str(e)}]"


class ConversationManager:
    """Manages dynamic conversations between agents with enhanced context"""

    def __init__(self):
        self.conversation_history: Dict[str, List[Dict]] = {}
        logger.info("Initialized ConversationManager")

    def start_conversation(
        self,
        agent1: SocialAgent,
        agent2: SocialAgent,
        num_turns: int = 4,
    ) -> str:
        """Initialize a conversation between two agents with relationship context"""
        try:
            conv_id = f"conv_{len(self.conversation_history)}_{random.randint(1000, 9999)}"
            self.conversation_history[conv_id] = []

            # Add conversation context
            is_friends = (
                agent1.persona.name in agent2.persona.friends
                or agent2.persona.name in agent1.persona.friends
            )
            relationship_type = (
                "friends" if is_friends else "new acquaintances"
            )

            context_entry = {
                "type": "context",
                "timestamp": datetime.datetime.now().isoformat(),
                "content": {
                    "agent1": {
                        "name": agent1.persona.name,
                        "description": agent1.persona.description,
                        "friends": agent1.persona.friends,
                    },
                    "agent2": {
                        "name": agent2.persona.name,
                        "description": agent2.persona.description,
                        "friends": agent2.persona.friends,
                    },
                    "relationship": relationship_type,
                },
            }
            self.conversation_history[conv_id].append(context_entry)

            logger.info(
                f"Starting conversation {conv_id} between {agent1.persona.name} and {agent2.persona.name} ({relationship_type})"
            )

            last_message = ""
            for turn in range(num_turns):
                # Agent 1's turn
                message1 = agent1.generate_message(
                    last_message, agent2.persona.name
                )
                self.add_message(
                    conv_id, agent1.persona.name, message1
                )
                last_message = message1

                # Agent 2's turn
                message2 = agent2.generate_message(
                    last_message, agent1.persona.name
                )
                self.add_message(
                    conv_id, agent2.persona.name, message2
                )
                last_message = message2

                logger.debug(
                    f"Completed turn {turn + 1}/{num_turns} of conversation {conv_id}"
                )

            return conv_id

        except Exception as e:
            logger.error(
                f"Error in conversation between {agent1.persona.name} and {agent2.persona.name}: {e}"
            )
            raise

    def add_message(
        self, conv_id: str, speaker_name: str, message: str
    ):
        """Add a message to the conversation history"""
        try:
            self.conversation_history[conv_id].append(
                {
                    "type": "message",
                    "speaker": speaker_name,
                    "message": message,
                    "timestamp": datetime.datetime.now().isoformat(),
                }
            )
            logger.debug(
                f"Added message from {speaker_name} to conversation {conv_id}"
            )
        except Exception as e:
            logger.error(
                f"Error adding message to conversation {conv_id}: {e}"
            )

    def print_conversation(self, conv_id: str):
        """Print a conversation with full context"""
        try:
            conversation = self.conversation_history[conv_id]
            context = next(
                (
                    entry
                    for entry in conversation
                    if entry["type"] == "context"
                ),
                None,
            )

            if context:
                print("\n=== Conversation Context ===")
                print(
                    f"Participant 1: {context['content']['agent1']['name']}"
                )
                print(
                    f"Description: {context['content']['agent1']['description']}"
                )
                print(
                    f"Friends: {', '.join(context['content']['agent1']['friends'])}\n"
                )

                print(
                    f"Participant 2: {context['content']['agent2']['name']}"
                )
                print(
                    f"Description: {context['content']['agent2']['description']}"
                )
                print(
                    f"Friends: {', '.join(context['content']['agent2']['friends'])}\n"
                )

                print(
                    f"Relationship: {context['content']['relationship']}"
                )
                print("=== Conversation Start ===\n")

            for entry in conversation:
                if entry["type"] == "message":
                    print(f"{entry['speaker']}: {entry['message']}\n")

        except Exception as e:
            logger.error(
                f"Error printing conversation {conv_id}: {e}"
            )


def run_social_simulation(
    num_agents: int = 10,
    num_conversations: int = 5,
    turns_per_conversation: int = 4,
):
    """Run the social simulation with specified parameters"""
    try:
        logger.info("Starting social simulation")

        network = DynamicSocialNetwork(
            api_key=os.getenv("OPENAI_API_KEY"),
            num_agents=num_agents,
        )

        # Start multiple conversations
        conversations = network.run_conversations(
            num_conversations, turns_per_conversation
        )
        logger.info(
            f"Completed social simulation with {len(conversations)} conversations"
        )

        # Print detailed conversation records
        for conv_id, initiator, partner in conversations:
            network.conversation_manager.print_conversation(conv_id)

        return conversations

    except Exception as e:
        logger.error(f"Error in social simulation: {e}")
        raise


if __name__ == "__main__":
    run_social_simulation()
