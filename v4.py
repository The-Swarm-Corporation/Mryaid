from loguru import logger
import os
import random
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import json
import datetime
import numpy as np
from datasets import load_dataset
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from swarms import Agent
from swarm_models import OpenAIChat

# Configure detailed logging
logger.remove()  # Remove default handler
logger.add(
    "social_simulation.log",
    rotation="500 MB",
    level="DEBUG",  # Changed to DEBUG for more detailed logging
    backtrace=True,  # Enable backtrace
    diagnose=True,  # Enable diagnosis
)

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
        try:
            # Print entry for debugging
            logger.info(f"Raw entry: {entry}")
            
            # Extract fields directly from the dataset entry
            synthesized_text = str(entry['synthesized_text']) if 'synthesized_text' in entry else ""
            input_persona = str(entry['input_persona']) if 'input_persona' in entry else ""
            description = str(entry['description']) if 'description' in entry else ""
            
            # Extract name from synthesized text
            name_match = re.search(r"Name:\s*([^,\n]+)", synthesized_text)
            name = name_match.group(1).strip() if name_match else "Unknown"
            
            logger.info(f"""
            Creating persona with:
            - Name: {name}
            - Description length: {len(description)}
            - Input persona length: {len(input_persona)}
            - Synthesized text length: {len(synthesized_text)}
            """)
            
            return cls(
                name=name,
                description=description,
                input_persona=input_persona,
                synthesized_text=synthesized_text
            )
            
        except Exception as e:
            logger.error(f"Error creating persona: {e}")
            logger.error(f"Entry that caused error: {entry}")
            raise ValueError(f"Failed to create persona from entry: {e}")


class LocalVectorStore:
    """Local vector store for persona matching using numpy"""

    def __init__(self):
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        self.personas: Dict[str, Persona] = {}
        self.embeddings: Dict[str, np.ndarray] = {}

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


class SocialAgent(Agent):
    """Enhanced agent with persona-based social capabilities"""

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
        out = f"""You are {self.persona.name}. {self.persona.description}

        Background: {self.persona.synthesized_text}
        
        Your current persona: {self.persona.input_persona}

        Maintain this persona while engaging in natural conversations. Follow these conversation stages:
        1. Start with a friendly greeting and ask how they are
        2. Share your name and ask for theirs
        3. Discuss what you do, based on your persona
        4. Engage in natural conversation about shared interests

        Always stay in character and respond as your persona would."""
        print(out)
        return out

    def generate_message(self, context: str = "", partner_name: str = "") -> str:
        """Generate a message based on context and awareness of conversation partner"""
        try:
            if not context:
                message = self.run(
                    f"Generate a friendly greeting for {partner_name} and ask how they are."
                )
            else:
                message = self.run(
                    f"{partner_name} said: '{context}'. Respond naturally as {self.persona.name}."
                )

            logger.info(
                f"Generated message from {self.persona.name}: {message[:50]}..."
            )
            return message

        except Exception as e:
            logger.error(
                f"Error generating message for {self.persona.name}: {e}"
            )
            return f"Hello! [Error: {str(e)}]"


class ConversationManager:
    """Manages dynamic conversations between agents"""

    def __init__(self):
        self.conversation_history: Dict[str, List[Dict]] = {}
        logger.info("Initialized ConversationManager")

    def start_conversation(
        self,
        agent1: SocialAgent,
        agent2: SocialAgent,
        num_turns: int = 4,
    ) -> str:
        """Initialize a conversation between two agents for specified number of turns"""
        try:
            conv_id = f"conv_{len(self.conversation_history)}_{random.randint(1000, 9999)}"
            self.conversation_history[conv_id] = []

            logger.info(
                f"Starting conversation {conv_id} between {agent1.persona.name} and {agent2.persona.name}"
            )

            last_message = ""
            for turn in range(num_turns):
                # Agent 1's turn
                message1 = agent1.generate_message(last_message, agent2.persona.name)
                self.add_message(
                    conv_id, agent1.persona.name, message1
                )
                last_message = message1

                # Agent 2's turn
                message2 = agent2.generate_message(last_message, agent1.persona.name)
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

    def export_conversations(
        self, filename: str = "conversation_history.json"
    ):
        """Export all conversations to JSON file"""
        try:
            with open(filename, "w") as f:
                json.dump(self.conversation_history, f, indent=2)
            logger.info(
                f"Exported conversation history to {filename}"
            )
        except Exception as e:
            logger.error(
                f"Error exporting conversations to {filename}: {e}"
            )


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
            logger.debug(f"First entry structure: {dataset['train'][0]}")
            logger.debug(f"First entry keys: {dataset['train'][0].keys()}")
            
            # Process entries
            for i, entry in enumerate(dataset['train']):
                if i >= num_agents:
                    break
                try:
                    logger.debug(f"Processing entry {i}:")
                    logger.debug(f"Entry type: {type(entry)}")
                    logger.debug(f"Entry content: {entry}")
                    
                    # Create persona with proper field access
                    persona = Persona(
                        name="Unknown",  # Will be updated from synthesized_text
                        description=str(entry.get('description', '')),
                        input_persona=str(entry.get('input_persona', '')),
                        synthesized_text=str(entry.get('synthesized_text', ''))
                    )
                    
                    # Try to extract name from synthesized text
                    name_match = re.search(r"Name:\s*([^,\n]+)", persona.synthesized_text)
                    if name_match:
                        persona.name = name_match.group(1).strip()
                    else:
                        persona.name = f"Persona_{i}"
                    
                    personas.append(persona)
                    logger.info(f"Successfully loaded persona: {persona.name}")
                    
                except Exception as e:
                    logger.error(f"Error loading individual persona {i}: {e}")
                    logger.error(f"Entry that caused error: {entry}")
                    continue
                    
            if not personas:
                logger.warning("No personas were successfully loaded!")
                
            logger.info(f"Successfully loaded {len(personas)} personas")
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

        return conversations

    except Exception as e:
        logger.error(f"Error in social simulation: {e}")
        raise


if __name__ == "__main__":
    logger.info("Starting main program")
    print(
        run_social_simulation(
            num_agents=4,
            num_conversations=5,
            turns_per_conversation=4,
        )
    )
