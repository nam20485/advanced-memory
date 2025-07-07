# **A Technical Report on Architecting and Implementing a Unified Knowledge and Memory Server using GraphRAG, Mem0, and the Model Context Protocol (MCP)**

## **Section 1: Foundational Concepts: The Synergy of Graph-based RAG and Agentic Memory**

The evolution of Large Language Models (LLMs) from stateless text generators to sophisticated, context-aware agents necessitates a parallel evolution in how they access and manage information. Standard Retrieval-Augmented Generation (RAG) was a foundational step, grounding LLM responses in external data to mitigate hallucinations and provide up-to-date information.1 However, this first-generation approach, often reliant on simple vector similarity search, struggles with complex queries that require synthesizing information across multiple documents or understanding nuanced relationships.3 Concurrently, the rise of agentic systems has highlighted the critical need for persistent memory, enabling agents to learn from past interactions, personalize responses, and maintain context over long periods—capabilities that stateless RAG systems lack.5

This report details the architecture and implementation of a next-generation system that addresses these limitations by integrating two powerful, complementary technologies: **GraphRAG** for deep, structured knowledge retrieval, and **Mem0** for stateful, agentic memory. This combination represents a significant architectural pattern, moving beyond simple question-answering to enable the creation of "digital experts"—AI agents that possess both a profound, structured understanding of a domain and a personalized, evolving memory of their interactions. The entire system is unified under the **Model Context Protocol (MCP)**, a standardized communication framework that ensures modularity and scalability.7

### **1.1. Deconstructing GraphRAG: From Unstructured Text to Actionable Knowledge Networks**

GraphRAG, a methodology pioneered by Microsoft Research, fundamentally re-imagines the retrieval process in RAG.3 Instead of treating a corpus of documents as a flat collection of text chunks for vector search, GraphRAG first transforms the unstructured text into a highly structured and interconnected knowledge graph. This paradigm shift allows for retrieval based not just on semantic similarity but on explicit, extracted relationships and the very structure of the knowledge itself.4

#### **The Indexing Pipeline: A Multi-Stage Transformation**

The core of GraphRAG's power lies in its offline indexing pipeline, a process that uses LLMs to build the knowledge graph before any queries are made.11 This pipeline typically involves several key stages:

1. **Text Chunking and Segmentation:** The process begins by breaking down large source documents (e.g.,.txt,.csv,.json files) into smaller, manageable "text units" or chunks. This is a necessary step to fit the content within the context window of the processing LLM.13 The choice of chunk size presents a trade-off: smaller chunks preserve fine-grained detail but increase processing costs, while larger chunks are more economical but may obscure specific facts.14  
2. **Entity and Relationship Extraction:** An LLM is prompted to analyze each text chunk to identify key entities—such as people, organizations, locations, or domain-specific concepts—and the relationships that connect them.10 The LLM extracts not only the entity names and types but also detailed descriptions and the nature of the relationship, including a numeric strength score in some implementations.14 This structured output, often in JSON format, forms the raw material for the graph.15  
3. **Knowledge Graph Construction:** The extracted entities and relationships are then aggregated into a single, unified knowledge graph. Entities become the nodes, and the relationships become the directed, weighted edges connecting them.3 This creates a rich, interconnected network that represents the holistic knowledge contained within the original document corpus.  
4. **Community Detection and Summarization:** To make sense of this potentially vast graph, GraphRAG employs graph machine learning algorithms, such as the Leiden community detection algorithm, to partition the graph into hierarchical clusters of densely connected entities.3 These "communities" represent emergent themes or semantic concepts within the data. An LLM is then used to generate a descriptive summary, or "community report," for each of these clusters from the bottom up. This creates a multi-layered, semantic hierarchy over the data, which is crucial for answering broad, thematic questions.16

#### **The Querying Advantage: Global and Local Search**

The structured, hierarchical nature of the indexed graph enables two distinct and powerful modes of querying that are not possible with standard vector-search RAG 3:

* **Global Search:** This mode is designed to answer high-level, abstract, or holistic questions about the entire dataset, such as "What are the main themes in this collection of reports?".16 Instead of searching through thousands of individual text chunks, the Global Search mechanism leverages the pre-generated community summaries. By processing these high-level summaries, the LLM can synthesize a comprehensive answer that captures the overarching structure and key topics of the corpus.3 More advanced implementations feature "dynamic" global search, which uses an LLM to traverse the community hierarchy, pruning irrelevant branches to improve both efficiency and the quality of the final answer.17  
* **Local Search:** This mode is optimized for answering specific questions about a particular entity, such as "What was Company X's involvement in Project Y?".16 When a query targets a specific entity, the Local Search retriever grounds itself on that entity's node in the graph. It then traverses the graph by "fanning out" to its neighbors, collecting information from connected entities and the relationships between them.11 This graph traversal provides rich, multi-hop context that is often missed by vector search, which may only find documents that mention both entities without explaining their connection. This ability to follow relationships is what allows GraphRAG to "connect the dots" across disparate pieces of information.4

### **1.2. The Agentic Memory Paradigm with Mem0: Achieving Stateful, Personalized Interactions**

While GraphRAG provides a powerful engine for understanding a static knowledge base, AI agents also require a mechanism for remembering and learning from dynamic interactions. This is the role of **Mem0**, a dedicated memory layer designed to transform stateless agents into stateful, adaptive partners that evolve over time.5

#### **Memory vs. RAG: A Critical Distinction**

It is essential to understand that agentic memory and RAG are complementary, not competing, technologies.5 RAG is fundamentally a stateless retrieval process; it excels at fetching factual information from a knowledge source but has no inherent knowledge of the user, their preferences, or the history of the conversation.5 Each query is treated as an isolated event.

Mem0, by contrast, provides **continuity**. It is designed to store the contextual fabric of interactions: user preferences, past decisions, conversation history, and learned facts about the user.5 This allows an agent to provide personalized, context-aware responses. For example, a RAG system can answer "What are some good vegetarian recipes?", but only an agent with Mem0 can remember "The user is a vegetarian and allergic to nuts" from a previous conversation and tailor its recommendations accordingly.

#### **A Multi-faceted Memory Model Inspired by Human Cognition**

Mem0 implements a sophisticated memory architecture that mirrors key aspects of human cognition, providing different types of memory for different purposes 5:

* **Working Memory:** Captures short-term, in-session awareness, keeping the agent focused on the immediate conversational context.5  
* **Episodic Memory:** Records specific past interactions and conversations, allowing the agent to recall entire exchanges or key moments from its history.5  
* **Factual Memory:** Stores long-term, structured knowledge, such as user preferences, settings, or important facts ("User's name is Alice," "User's favorite hobby is tennis").5  
* **Semantic Memory:** Builds generalized knowledge over time, abstracting patterns and concepts from multiple interactions.5

#### **The Hybrid Storage and Retrieval Mechanism**

To support this rich memory model, Mem0 utilizes a hybrid backend storage architecture. When a memory is added, an LLM intelligently extracts key information and stores it across multiple data stores, including a vector database for efficient semantic search, a key-value store, and, crucially, a **graph database**.19 The use of a graph database (such as Neo4j or Memgraph) is particularly noteworthy, as it allows Mem0 to create and track relationships

*between* memories, connecting insights and entities across different sessions.22 This enables more nuanced retrieval that goes beyond simple semantic similarity, uncovering connections in the agent's own experience. Retrieval is a scored process that considers relevance, importance, and recency to ensure only the most useful context is surfaced, optimizing token usage and reducing latency.5

### **1.3. The Architectural Blueprint: A Symbiotic Model for Knowledge and Memory**

The integration of GraphRAG and Mem0 creates a symbiotic architecture that is far more capable than either component in isolation. This system can be conceptualized as providing an AI agent with two distinct but interconnected cognitive functions, akin to a human expert.

#### **The "Two Brains" Analogy**

* **GraphRAG as the "Long-Term Semantic Knowledge Base":** This component acts as the agent's deep, structured library of domain expertise. It is analogous to a doctor's medical training or a lawyer's knowledge of case law. This knowledge is comprehensive, highly organized, and relatively static, updated through periodic, intensive indexing processes.3 It answers the question, "What is known about subject X in the established body of knowledge?"  
* **Mem0 as the "Personal Episodic & Working Memory":** This component serves as the agent's evolving, personalized understanding of its interactions with a specific user. It is analogous to a doctor's memory of a particular patient's history, allergies, and past treatments. This memory is dynamic, personal, and built incrementally with every new interaction.5 It answers the question, "What do I know about  
  *this user* and our past conversations regarding subject X?"

This dual-component design mirrors a fundamental convergence in AI systems development. The first trend is the move toward structuring vast amounts of unstructured data to enable more intelligent retrieval, perfectly embodied by GraphRAG. The second is the push for stateful, long-running agents that can build relationships and personalize experiences, a need directly addressed by Mem0. The combination of these two trends is not merely an interesting technical exercise; it is a necessary step toward creating truly useful AI assistants for complex, high-stakes domains like financial advising, legal research, or personalized healthcare, where both deep domain knowledge and an understanding of individual context are non-negotiable.1

#### **The "Graph-of-Graphs" Architecture: A Deeper Connection**

A more nuanced examination of the underlying technologies reveals a fascinating "graph-of-graphs" architectural pattern. GraphRAG's primary output is a large, pre-computed knowledge graph representing the "objective" knowledge space of the document corpus. Simultaneously, Mem0, when configured with a graph backend like Neo4j, dynamically builds its own graph of memories, representing the "subjective" experience space of the agent's interactions.22

This creates a system with two distinct but related graph structures. The true potential of this architecture lies in the future possibility of creating explicit links *between* these two graphs. For instance, a memory node in the Mem0 graph (e.g., "User asked about 'dynamic community selection'") could be programmatically linked to the corresponding community node within the GraphRAG knowledge graph. This would create an incredibly rich, multi-layered context, allowing the agent to seamlessly navigate between its personal experience and the foundational knowledge base, leading to unprecedented levels of reasoning and contextual understanding.

## **Section 2: System Architecture: Designing the MCP-based Knowledge and Memory Tool**

With the foundational concepts established, the next step is to design a coherent system architecture that allows an AI agent to effectively leverage both the GraphRAG knowledge core and the Mem0 memory layer. The Model Context Protocol (MCP) provides the ideal framework for this integration, enabling a modular, scalable, and standardized approach to building agentic tools.

### **2.1. The Model Context Protocol (MCP) as the Unifying Framework**

The Model Context Protocol (MCP) is an emerging standard designed to facilitate communication between AI agents and external tools or services.7 It acts as a universal adapter, decoupling the agent's core reasoning logic from the specific implementation details of the tools it uses. This is a crucial design principle for building robust and maintainable AI systems.

MCP is particularly well-suited for this architecture because it is designed with "cloud-native" principles in mind, where the agent and its tools can operate as independent, decoupled processes.7 This allows the GraphRAG/Mem0 server to be developed, scaled, and maintained separately from the agent itself. While MCP supports various transport mechanisms, this report will focus on using

**Server-Sent Events (SSE)**, a simple and efficient protocol for pushing real-time updates from a server to a client. This approach is demonstrated in reference implementations like mcp-mem0 and is ideal for the asynchronous, request-response nature of agent-tool interactions.7

### **2.2. High-Level System Diagram and Component Interaction**

The proposed system operates through a clear and logical flow of data and requests, orchestrated by the MCP server. The interaction between components can be visualized as follows:

1. **User-Agent Interaction:** The process begins with a user sending a prompt or query to the AI Agent.  
2. **Agent Reasoning and Tool Selection:** The AI Agent (e.g., one built with a framework like LangGraph or CrewAI) processes the user's input. Its internal reasoning model determines that it needs external information to formulate a high-quality response and decides which specific tool is required (e.g., searching the knowledge base vs. searching personal memory).  
3. **MCP Request:** The Agent constructs an MCP-compliant request, specifying the tool to be called and its parameters. This request is sent to the MCP Server's designated SSE endpoint (e.g., http://localhost:8080/sse).  
4. **MCP Server Routing:** The MCP Server, a FastAPI application in this implementation, receives the request. It parses the tool call and routes it to the corresponding Python function that implements the tool's logic.  
5. **Tool Execution (Knowledge or Memory):**  
   * If a **knowledge query** is requested (e.g., query\_knowledge\_base), the server invokes the GraphRAGKnowledgeProvider module. This module, in turn, interacts with the chosen GraphRAG engine (Microsoft, Neo4j, or LlamaIndex) and its underlying data stores (Parquet files, a Neo4j database, etc.) to retrieve the relevant information.  
   * If a **memory operation** is requested (e.g., add\_interaction\_memory), the server invokes the Mem0MemoryProvider module. This module uses the Mem0 Python SDK to communicate with the Mem0 backend (whether self-hosted or the managed platform), performing the requested action (add, search, update, etc.).  
6. **MCP Response:** The result from the tool execution is formatted into a standardized MCP response.  
7. **Response to Agent:** The MCP Server sends this response back to the AI Agent through the open SSE connection.  
8. **Final Generation:** The Agent incorporates the retrieved knowledge and/or memory context into its prompt and uses its LLM to generate the final, context-aware response for the user.

### **2.3. Defining the Toolset: A Granular API for Knowledge and Memory**

The design of the API that the MCP server exposes to the agent is a critical architectural decision. A granular, well-defined toolset is superior to a single, monolithic search tool because it empowers the agent to make more explicit and intelligent decisions about the *type* of information it needs. This leads to better performance, improved observability, and more accurate results. By creating distinct tools, the agent's "intent" is mapped directly to the underlying system's specialized capabilities. This makes the agent's chain-of-thought more transparent and debuggable, as a developer can verify whether the agent correctly chose a global search for a thematic question or a local search for a specific one.20

The following API contract is proposed, synthesizing the capabilities of both GraphRAG and Mem0 into a logical set of tools:

#### **Knowledge Tool (Powered by GraphRAG)**

This tool provides access to the deep, structured domain knowledge indexed by the GraphRAG engine.

* **query\_knowledge\_base(query: str, search\_type: Literal\['global', 'local'\], user\_context: Optional\[str\] \= None) \-\> str**  
  * **Description:** Queries the main knowledge base to answer domain-specific questions.  
  * **Parameters:**  
    * query: str: The user's natural language question.  
    * search\_type: Literal\['global', 'local'\]: A mandatory parameter that forces the agent to decide on the optimal search strategy. 'global' is used for broad, thematic questions, while 'local' is for specific queries about known entities.16  
    * user\_context: Optional\[str\]: An optional string providing context about the user, typically derived from a preliminary call to the memory tools. This is the explicit bridge between the memory and knowledge systems, allowing personalization to influence knowledge retrieval (e.g., user\_context="The user is an expert in this field, provide a technical answer.").  
  * **Returns:** A string containing the synthesized answer from the knowledge base.

#### **Memory Tools (Powered by Mem0)**

These tools manage the agent's personal, evolving memory of its interactions with each user.

* **add\_interaction\_memory(user\_id: str, conversation\_turn: List, metadata: Optional \= None) \-\> Dict**  
  * **Description:** Stores a recent conversation turn into the user's long-term memory. This should be called after each meaningful user-agent exchange.  
  * **Parameters:**  
    * user\_id: str: The unique identifier for the user whose memory is being updated. This is essential for multi-user personalization.22  
    * conversation\_turn: List: The list of user and assistant messages (e.g., \[{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}\]) to be processed and stored.23  
    * metadata: Optional: Optional key-value pairs to tag the memory (e.g., {"topic": "financial\_planning"}).  
  * **Returns:** A dictionary confirming the status of the operation.  
* **search\_user\_memory(user\_id: str, query: str, limit: int \= 5\) \-\> List**  
  * **Description:** Performs a semantic search over a specific user's memory to retrieve relevant past interactions, preferences, or facts.  
  * **Parameters:**  
    * user\_id: str: The user whose memory to search.  
    * query: str: The natural language query to find relevant memories (e.g., "What are my preferences?").  
    * limit: int: The maximum number of memory items to return.  
  * **Returns:** A list of dictionaries, where each dictionary represents a retrieved memory.22  
* **get\_user\_profile(user\_id: str) \-\> Dict**  
  * **Description:** A composite tool that retrieves key memories (preferences, stated goals, personal facts) and synthesizes them into a concise user profile. This is useful for quickly grounding the agent at the beginning of a new session.  
  * **Parameters:**  
    * user\_id: str: The user for whom to generate a profile.  
  * **Returns:** A dictionary summarizing the user's key attributes as understood by the agent.

This well-defined toolset forms the bedrock of the MCP server, providing the agent with a clear and powerful interface to its extended cognitive faculties.

## **Section 3: Implementation Deep Dive: Building the GraphRAG Knowledge Core**

Implementing the GraphRAG component is a pivotal step that involves transforming a raw corpus of documents into a queryable knowledge graph. There is no single, universal "GraphRAG library"; rather, it is a methodology with several powerful and distinct implementations. The choice of which engine to use is a key architectural decision that depends on factors such as the desired level of control, existing technology stack (e.g., use of Neo4j), and the need for integration with broader AI frameworks like LlamaIndex or LangChain.9 This section provides a comparative analysis and practical implementation guide for three primary paths.

### **3.1. Choosing Your GraphRAG Engine: A Comparative Analysis**

Before writing any code, it is essential to select the implementation path that best aligns with the project's goals. Each approach offers a different set of trade-offs between ease of use, control, and backend architecture. The following table provides a high-level comparison to guide this decision.

**Table 3.1: Comparative Analysis of GraphRAG Implementation Approaches**

| Criteria | Microsoft graphrag Library | neo4j-graphrag Library | LlamaIndex / LangChain |
| :---- | :---- | :---- | :---- |
| **Primary Interaction Model** | CLI & YAML Configuration-driven 11 | Native Python SDK 29 | High-level Framework Abstractions 15 |
| **Underlying Storage** | Parquet files for graph data and artifacts 30 | Native Neo4j Graph Database 29 | In-memory or pluggable Graph/Vector Stores 15 |
| **Ease of Programmatic Integration** | Moderate; requires wrapping CLI commands or using internal functions 33 | High; designed for programmatic use with a clear SDK 35 | Very High; seamlessly integrates into existing LlamaIndex/LangChain pipelines 15 |
| **Level of Control/Customization** | High; through detailed YAML configuration and prompt tuning 9 | Very High; through custom Cypher queries, retriever classes, and pipeline components 31 | Medium to High; through component swapping and custom classes within the framework 15 |
| **Key Dependency** | Python Environment, LLM API Key 37 | Neo4j Database (AuraDB or self-hosted) 29 | LlamaIndex or LangChain Framework 16 |
| **Best For** | Reproducing original research; deep customization of the core pipeline; projects comfortable with a config-driven workflow. | Production systems leveraging a true graph database; applications requiring complex graph traversals and Cypher queries. | Rapid prototyping; projects already invested in the LlamaIndex/LangChain ecosystem; applications needing modularity. |

### **3.2. Path A: Implementation with Microsoft's graphrag Library**

This is the reference implementation from Microsoft Research and offers the most direct access to the original methodology.9 It is primarily driven by command-line interface (CLI) commands and a central

settings.yaml file, which makes programmatic integration less direct but still achievable.

#### **Indexing**

The indexing process is typically run from the command line. However, for an automated server, it is necessary to trigger this process programmatically. This can be done by importing and calling the internal CLI function directly.34

**Setup:**

1. **Install the library:** pip install graphrag.11  
2. **Create a project directory:**  
   Bash  
   mkdir \-p graphrag\_project/input

3. **Add data:** Place your raw text files (e.g., doc1.txt, doc2.txt) into the graphrag\_project/input directory.11  
4. **Initialize configuration:** Run the init command from within the graphrag\_project directory.  
   Bash  
   python \-m graphrag.index \--init \--root.

   This creates a settings.yaml file and a .env file.30  
5. **Configure:** Edit .env to add your GRAPHRAG\_API\_KEY. In settings.yaml, configure your LLM model, embedding model, and other pipeline settings as needed.30

**Programmatic Indexing Code:**

The following Python script demonstrates how to run the indexing pipeline programmatically.

Python

import os  
from graphrag.index.cli import index\_cli

def run\_microsoft\_graphrag\_indexing(project\_root: str):  
    """  
    Programmatically runs the Microsoft GraphRAG indexing pipeline.

    Args:  
        project\_root: The absolute path to the GraphRAG project root directory.  
    """  
    print(f"Starting GraphRAG indexing for project at: {project\_root}")  
      
    \# The index\_cli function is the programmatic entry point for the indexing command.  
    \# We set init=False because we assume the config files already exist.  
    try:  
        index\_cli(  
            root=project\_root,  
            init=False,  
            verbose=True,  
            resume=None,  
            \# Other parameters can be set here as needed  
        )  
        print("GraphRAG indexing completed successfully.")  
        print(f"Output artifacts can be found in: {os.path.join(project\_root, 'output')}")  
    except Exception as e:  
        print(f"An error occurred during indexing: {e}")

if \_\_name\_\_ \== '\_\_main\_\_':  
    \# Ensure you use an absolute path for reliability  
    project\_path \= os.path.abspath("./graphrag\_project")  
    run\_microsoft\_graphrag\_indexing(project\_path)

#### **Querying**

Querying with the Microsoft library is also CLI-based. To integrate this into a Python server, the most robust method is to wrap the CLI command in a Python subprocess, as demonstrated in a community example for creating a FastAPI wrapper.33

**Programmatic Querying Code:**

Python

import subprocess  
import json

def query\_microsoft\_graphrag(project\_root: str, query: str, method: str \= "global") \-\> dict:  
    """  
    Programmatically queries the indexed GraphRAG data using a subprocess.

    Args:  
        project\_root: The path to the GraphRAG project root directory.  
        query: The natural language query.  
        method: The search method, either 'global' or 'local'.

    Returns:  
        A dictionary containing the response and any errors.  
    """  
    if method not in \["global", "local"\]:  
        return {"error": "Invalid search method. Must be 'global' or 'local'."}

    command \= \[  
        "python",  
        "-m",  
        "graphrag.query",  
        "--root",  
        project\_root,  
        "--method",  
        method,  
        query,  
    \]

    print(f"Executing GraphRAG query: {' '.join(command)}")  
      
    try:  
        process \= subprocess.run(  
            command,  
            capture\_output=True,  
            text=True,  
            check=True,  
            cwd=project\_root  
        )  
          
        \# The output from the query command is typically a JSON-like string.  
        \# It needs to be parsed to extract the actual answer.  
        response\_text \= process.stdout  
        \# Basic parsing, may need to be more robust for production  
        answer \= response\_text.split("Response:").strip()

        return {  
            "answer": answer,  
            "stdout": process.stdout,  
            "stderr": process.stderr  
        }  
    except subprocess.CalledProcessError as e:  
        print(f"Error executing GraphRAG query: {e}")  
        return {  
            "error": "Query execution failed.",  
            "stdout": e.stdout,  
            "stderr": e.stderr  
        }

if \_\_name\_\_ \== '\_\_main\_\_':  
    project\_path \= "./graphrag\_project"  
      
    \# Example Global Search  
    global\_query \= "What are the top themes in the documents?"  
    global\_response \= query\_microsoft\_graphrag(project\_path, global\_query, "global")  
    print("\\n--- Global Search Response \---")  
    print(json.dumps(global\_response, indent=2))

    \# Example Local Search  
    local\_query \= "Tell me more about the entity 'Albert Einstein'."  
    local\_response \= query\_microsoft\_graphrag(project\_path, local\_query, "local")  
    print("\\n--- Local Search Response \---")  
    print(json.dumps(local\_response, indent=2))

### **3.3. Path B: Implementation with neo4j-graphrag for a Native Graph Backend**

This is the official, actively maintained package from Neo4j and is the recommended path for production systems that can benefit from a dedicated graph database.29 It provides a native Python SDK for both indexing and querying.

#### **Indexing (Knowledge Graph Builder)**

The neo4j-graphrag library provides a SimpleKGPipeline for easily transforming documents into a knowledge graph and loading it into Neo4j.38

**Setup:**

1. **Install dependencies:** pip install "neo4j-graphrag\[openai\]".29  
2. **Set up Neo4j:** Use a free Neo4j AuraDB instance or run Neo4j in Docker. Get your URI, username, and password.31  
3. **Set Environment Variables:** Set OPENAI\_API\_KEY, NEO4J\_URI, NEO4J\_USERNAME, and NEO4J\_PASSWORD.

**Programmatic Indexing Code:**

Python

import os  
from neo4j import GraphDatabase  
from neo4j\_graphrag.llm import OpenAI  
from neo4j\_graphrag.embedder import OpenAIEmbedder  
from neo4j\_graphrag.pipeline import SimpleKGPipeline

\# Load credentials from environment variables  
NEO4J\_URI \= os.getenv("NEO4J\_URI")  
NEO4J\_USERNAME \= os.getenv("NEO4J\_USERNAME")  
NEO4J\_PASSWORD \= os.getenv("NEO4J\_PASSWORD")  
OPENAI\_API\_KEY \= os.getenv("OPENAI\_API\_KEY")

\# Sample documents to be indexed  
documents \=

def run\_neo4j\_graphrag\_indexing(docs: list):  
    """  
    Uses SimpleKGPipeline to build a knowledge graph in Neo4j.  
    """  
    print("Starting Neo4j GraphRAG indexing...")  
      
    \# Define the LLM and Embedder for the pipeline  
    llm \= OpenAI(api\_key=OPENAI\_API\_KEY, model\_name="gpt-4o-mini")  
    embedder \= OpenAIEmbedder(api\_key=OPENAI\_API\_KEY)

    \# Define the structure of the graph to be extracted  
    node\_types \=  
    relationship\_types \=  
      
    \# Initialize the pipeline  
    kg\_pipeline \= SimpleKGPipeline(  
        llm=llm,  
        embedder=embedder,  
        node\_types=node\_types,  
        relationship\_types=relationship\_types  
    )  
      
    \# Connect to Neo4j  
    driver \= GraphDatabase.driver(NEO4J\_URI, auth=(NEO4J\_USERNAME, NEO4J\_PASSWORD))  
      
    try:  
        \# Run the pipeline on the documents  
        kg\_pipeline.run(docs, driver=driver)  
        print("Neo4j GraphRAG indexing completed successfully.")  
        print("Graph data has been loaded into your Neo4j database.")  
    except Exception as e:  
        print(f"An error occurred during Neo4j indexing: {e}")  
    finally:  
        driver.close()

if \_\_name\_\_ \== '\_\_main\_\_':  
    run\_neo4j\_graphrag\_indexing(documents)

#### **Querying**

Querying with neo4j-graphrag involves using the GraphRAG class, which orchestrates a retriever and an LLM. The most powerful retriever is the VectorCypherRetriever, which combines semantic search with a custom graph traversal Cypher query.31

**Programmatic Querying Code:**

Python

import os  
from neo4j import GraphDatabase  
from neo4j\_graphrag.llm import OpenAILLM  
from neo4j\_graphrag.embeddings import OpenAIEmbeddings  
from neo4j\_graphrag.retrievers import VectorCypherRetriever  
from neo4j\_graphrag.generation import GraphRAG

\# Load credentials  
NEO4J\_URI \= os.getenv("NEO4J\_URI")  
NEO4J\_USERNAME \= os.getenv("NEO4J\_USERNAME")  
NEO4J\_PASSWORD \= os.getenv("NEO4J\_PASSWORD")  
OPENAI\_API\_KEY \= os.getenv("OPENAI\_API\_KEY")

\# The name of the vector index created on the 'Chunk' nodes' embedding property  
INDEX\_NAME \= "text\_embeddings"

def query\_neo4j\_graphrag(query\_text: str) \-\> str:  
    """  
    Performs a RAG query using the neo4j-graphrag library.  
    """  
    print(f"Querying Neo4j GraphRAG with: '{query\_text}'")  
      
    driver \= GraphDatabase.driver(NEO4J\_URI, auth=(NEO4J\_USERNAME, NEO4J\_PASSWORD))  
      
    \# 1\. Initialize Embedder and LLM  
    embedder \= OpenAIEmbeddings(api\_key=OPENAI\_API\_KEY, model="text-embedding-3-small")  
    llm \= OpenAILLM(api\_key=OPENAI\_API\_KEY, model\_name="gpt-4o-mini")

    \# 2\. Define the graph traversal query  
    \# This Cypher query starts from a Chunk found via vector search (\`node\`)  
    \# and traverses 1 to 2 hops out in the entity graph to find related context.  
    retrieval\_query \= """  
    MATCH (node)\<--(e)  
    CALL {  
        WITH e  
        MATCH p=(e)-\[\*1..2\]-(neighbor)  
        RETURN p  
    }  
    RETURN e.text AS text, score,  
           \[rel in relationships(p) |   
            {  
                start: startNode(rel).id,  
                type: type(rel),  
                end: endNode(rel).id  
            }  
           \] AS graph  
    """

    \# 3\. Initialize the VectorCypherRetriever  
    retriever \= VectorCypherRetriever(  
        driver,  
        index\_name=INDEX\_NAME,  
        retrieval\_query=retrieval\_query,  
        embedder=embedder  
    )

    \# 4\. Initialize the GraphRAG pipeline  
    rag\_pipeline \= GraphRAG(retriever=retriever, llm=llm)

    \# 5\. Execute the search  
    try:  
        response \= rag\_pipeline.search(  
            query\_text=query\_text,  
            retriever\_config={"top\_k": 3}  
        )  
        return response.answer  
    except Exception as e:  
        return f"An error occurred during query: {e}"  
    finally:  
        driver.close()

if \_\_name\_\_ \== '\_\_main\_\_':  
    \# First, ensure the index exists (can be created with the pipeline or manually)  
    \# from neo4j\_graphrag.indexes import create\_vector\_index  
    \# driver \= GraphDatabase.driver(NEO4J\_URI, auth=(NEO4J\_USERNAME, NEO4J\_PASSWORD))  
    \# create\_vector\_index(driver, INDEX\_NAME, label="Chunk", embedding\_property="embedding", dimensions=1536)  
    \# driver.close()

    question \= "Who developed the theory of relativity and where was he born?"  
    answer \= query\_neo4j\_graphrag(question)  
    print("\\n--- Query Response \---")  
    print(answer)

### **3.4. Path C: Implementation with LlamaIndex for a Modular, Extensible Approach**

LlamaIndex offers a high-level, component-based framework for building complex RAG systems, including GraphRAG.15 This path is excellent for developers who value modularity and want to integrate GraphRAG into a larger LlamaIndex application. The implementation involves creating custom components for extraction and storage, as detailed in the official cookbook.15

#### **Indexing and Querying**

The LlamaIndex approach combines indexing and querying into a more unified pipeline centered around the PropertyGraphIndex. The following code is a condensed version of the detailed LlamaIndex cookbook, demonstrating the end-to-end process.15

**Setup:**

1. **Install dependencies:** pip install llama-index llama-index-llms-openai graspologic  
2. **Set Environment Variables:** Set OPENAI\_API\_KEY.

**End-to-End LlamaIndex GraphRAG Code:**

Python

import os  
import re  
import json  
import nest\_asyncio  
from typing import List, Any, Callable, Optional, Union  
import networkx as nx  
from graspologic.partition import hierarchical\_leiden

from llama\_index.core import Document, Settings, PropertyGraphIndex  
from llama\_index.core.graph\_stores import SimplePropertyGraphStore  
from llama\_index.core.llms import ChatMessage  
from llama\_index.llms.openai import OpenAI  
from llama\_index.core.node\_parser import SentenceSplitter  
from llama\_index.core.query\_engine import CustomQueryEngine  
from llama\_index.core.schema import TransformComponent, BaseNode  
from llama\_index.core.indices.property\_graph.utils import default\_parse\_triplets\_fn  
from llama\_index.core.graph\_stores.types import EntityNode, Relation, KG\_NODES\_KEY, KG\_RELATIONS\_KEY

\# Apply nest\_asyncio to run async code in a Jupyter-like environment  
nest\_asyncio.apply()

\# \--- Custom Components from LlamaIndex Cookbook \[15\] \---

\# \[15\]

\# Placeholder for the full class definitions. In a real script, these would be included.  
class GraphRAGExtractor(TransformComponent):  
    \# Full implementation from \[15\] goes here  
    def \_\_init\_\_(self, llm, extract\_prompt, parse\_fn, \*\*kwargs): self.llm \= llm; self.extract\_prompt \= extract\_prompt; self.parse\_fn \= parse\_fn; self.max\_paths\_per\_chunk \= kwargs.get('max\_paths\_per\_chunk', 10)  
    async def \_aextract(self, node: BaseNode):  
        text \= node.get\_content(metadata\_mode="llm")  
        try:  
            llm\_response \= await self.llm.apredict(self.extract\_prompt, text=text, max\_knowledge\_triplets=self.max\_paths\_per\_chunk)  
            entities, relationships \= self.parse\_fn(llm\_response)  
        except Exception: entities, relationships \=,  
        node.metadata \= \[EntityNode(name=e, label=e, properties={"description": e}) for e in entities\]  
        node.metadata \=, target\_id=r, label=r, properties={"description": r}) for r in relationships\]  
        return node  
    async def acall(self, nodes: List, \*\*kwargs): return \[await self.\_aextract(n) for n in nodes\]

class GraphRAGStore(SimplePropertyGraphStore):  
    \# Full implementation from \[15\] goes here  
    def build\_communities(self): pass \# Placeholder  
    def get\_community\_summaries(self): return {0: "Placeholder community summary"} \# Placeholder

class GraphRAGQueryEngine(CustomQueryEngine):  
    \# Full implementation from \[15\] goes here  
    def custom\_query(self, query\_str: str): return "Placeholder answer from LlamaIndex GraphRAG."

def custom\_parse\_fn(response\_str: str) \-\> Any:  
    \# A robust JSON parser for the LLM output  
    json\_pattern \= r"\`\`\`json\\s\*(\\{.\*?\\})\\s\*\`\`\`"  
    match \= re.search(json\_pattern, response\_str, re.DOTALL)  
    if not match:  
        match \= re.search(r"(\\{.\*?\\})", response\_str, re.DOTALL)  
    if not match: return,  
    json\_str \= match.group(1)  
    try:  
        data \= json.loads(json\_str)  
        entities \= \[(e\["entity\_name"\], e\["entity\_type"\], e.get("entity\_description", "")) for e in data.get("entities",)\]  
        relationships \= \[(r\["source\_entity"\], r\["target\_entity"\], r\["relation"\], r.get("relationship\_description", "")) for r in data.get("relationships",)\]  
        return entities, relationships  
    except json.JSONDecodeError: return,  
      
\# \--- End-to-End Pipeline \---

def run\_llamaindex\_graphrag\_pipeline(docs: List, query: str):  
    """  
    Builds and queries a GraphRAG system using LlamaIndex.  
    """  
    print("Starting LlamaIndex GraphRAG pipeline...")  
      
    \# 1\. Configure LLM  
    Settings.llm \= OpenAI(model="gpt-4o-mini")  
      
    \# 2\. Split documents into nodes  
    splitter \= SentenceSplitter(chunk\_size=512)  
    nodes \= splitter.get\_nodes\_from\_documents(docs)  
      
    \# 3\. Define the extraction prompt  
    \# Using a simplified prompt for this example  
    kg\_extract\_prompt \= (  
        "Extract entities and relationships from the text below. "  
        "Return a JSON object with 'entities' and 'relationships' keys.\\n"  
        "Text: {text}\\n"  
        "JSON Output:"  
    )

    \# 4\. Initialize the GraphRAGExtractor  
    kg\_extractor \= GraphRAGExtractor(  
        llm=Settings.llm,  
        extract\_prompt=kg\_extract\_prompt,  
        parse\_fn=custom\_parse\_fn,  
        max\_paths\_per\_chunk=5  
    )

    \# 5\. Build the PropertyGraphIndex  
    \# This step runs the extraction and builds the graph in the GraphRAGStore  
    graph\_store \= GraphRAGStore()  
    index \= PropertyGraphIndex(  
        nodes=nodes,  
        property\_graph\_store=graph\_store,  
        kg\_extractors=\[kg\_extractor\],  
        show\_progress=True,  
    )  
      
    \# 6\. Build communities (a key step in GraphRAG)  
    \# The custom GraphRAGStore would handle this logic  
    graph\_store.build\_communities()

    \# 7\. Create the Query Engine and execute the query  
    query\_engine \= GraphRAGQueryEngine(  
        graph\_store=graph\_store,  
        llm=Settings.llm  
    )  
      
    response \= query\_engine.query(query)  
    return response

if \_\_name\_\_ \== '\_\_main\_\_':  
    \# Set API Key  
    os.environ\["OPENAI\_API\_KEY"\] \= os.getenv("OPENAI\_API\_KEY")

    \# Sample documents  
    sample\_docs \=  
      
    question \= "What is the relationship between Elon Musk and SpaceX?"  
      
    \# Note: The custom classes are placeholders. A full run requires the complete  
    \# implementations from the LlamaIndex cookbook.\[15\]  
    print("Note: This is a demonstrative script. Full execution requires the complete custom class definitions.")  
    \# answer \= run\_llamaindex\_graphrag\_pipeline(sample\_docs, question)  
    \# print("\\n--- LlamaIndex GraphRAG Response \---")  
    \# print(answer)

This comprehensive analysis of the three main implementation paths provides the necessary detail for an engineer to make an informed choice and begin building the knowledge core of the agentic system.

## **Section 4: Implementation Deep Dive: Integrating the Mem0 Agentic Memory Layer**

After establishing the knowledge core with GraphRAG, the next critical component is the agent's memory. Mem0 provides a sophisticated, easy-to-integrate memory layer that enables personalization and statefulness.5 This section offers a complete guide to setting up and programmatically interacting with Mem0 using its Python SDK.

### **4.1. Mem0 Setup: Self-Hosted vs. Managed Platform**

Mem0 offers two deployment models, allowing developers to choose between a fully managed service for convenience and an open-source solution for maximum control and self-hosting.5

* **Managed Platform:** This is the quickest way to get started. It operates as a SaaS solution where memory management, data storage, and infrastructure are handled by Mem0. Integration is achieved simply by signing up on the Mem0 platform, obtaining an API key, and using it to initialize the client in your application.28 This path is ideal for teams that want to focus on agent logic without managing backend infrastructure and benefit from features like advanced analytics and enterprise security.28  
* **Self-Hosted (Open Source):** For developers who require data to remain within their own infrastructure or desire full control over the environment, the open-source mem0ai package is the solution.22 This approach requires setting up the necessary backend dependencies yourself. At a minimum, a vector database is needed for semantic search. For advanced relationship tracking between memories, a graph database can also be configured.24 A common setup involves using Docker to run instances of Qdrant (for vectors) and Neo4j (for graphs).41  
  **Example Self-Hosted Setup with Docker:**  
  Bash  
  \# Pull and run Qdrant for vector storage  
  docker pull qdrant/qdrant  
  docker run \-p 6333:6333 \-p 6334:6334 \\  
      \-v $(pwd)/qdrant\_storage:/qdrant/storage:z \\  
      qdrant/qdrant

  \# Pull and run Neo4j for graph storage (optional but recommended)  
  docker run \-p 7474:7474 \-p 7687:7687 \\  
      \-v $(pwd)/neo4j/data:/data \\  
      \-e NEO4J\_AUTH=neo4j/password \\  
      neo4j:latest

### **4.2. The Mem0 Python SDK: Core API in Practice**

The mem0ai Python package provides a clean and intuitive API for all memory operations. The following examples demonstrate the core functionalities required for the MCP server.

**Installation:**

Bash

pip install "mem0ai\[openai,graph\]"

This command installs the core library along with optional dependencies for OpenAI integration and graph database support.24

**Initialization:**

The Memory class is the main entry point. Its initialization depends on the chosen setup.

Python

import os  
from mem0 import Memory

\# Set your LLM API key  
os.environ\["OPENAI\_API\_KEY"\] \= "your-openai-api-key"

\# \--- Option 1: Basic in-memory (for quick tests) \---  
\# Data is not persisted across runs.  
m\_in\_memory \= Memory()  
print("Initialized basic in-memory Mem0 instance.")

\# \--- Option 2: Self-hosted with configuration \---  
\# Connects to the Docker containers set up previously.  
config\_self\_hosted \= {  
    "vector\_store": {  
        "provider": "qdrant",  
        "config": {  
            "host": "localhost",  
            "port": 6333,  
        }  
    },  
    "graph\_store": { \# Optional: for graph-based memory  
        "provider": "neo4j",  
        "config": {  
            "url": "bolt://localhost:7687",  
            "username": "neo4j",  
            "password": "password"  
        }  
    }  
}  
m\_self\_hosted \= Memory.from\_config(config\_self\_hosted)  
print("Initialized self-hosted Mem0 instance with Qdrant and Neo4j.")

\# \--- Option 3: Managed Platform using MemoryClient \---  
\# Requires an API key from app.mem0.ai  
from mem0 import MemoryClient  
os.environ\["MEM0\_API\_KEY"\] \= "your-mem0-api-key"  
client\_managed \= MemoryClient()  
print("Initialized Mem0 client for managed platform.")

**Core Memory Operations:**

The following code snippets illustrate the primary functions for managing memories, using the m\_self\_hosted instance as an example. These functions directly map to the tools defined in the MCP server architecture.

Python

\# A unique ID for the user is required for all operations  
user\_id \= "alice123"

\# 1\. Add Memory: Storing information  
\# Mem0 can process unstructured text or structured conversation lists.  
conversation \=  
result\_add \= m\_self\_hosted.add(  
    data=conversation,   
    user\_id=user\_id,   
    metadata={"topic": "hobbies", "activity": "hiking"}  
)  
print(f"Add memory result: {result\_add}")

\# 2\. Search Memory: Retrieving relevant information  
query \= "What are my hobbies?"  
related\_memories \= m\_self\_hosted.search(query=query, user\_id=user\_id, limit=3)  
print(f"Search results for '{query}':")  
for mem in related\_memories:  
    print(f"  \- ID: {mem\['id'\]}, Memory: {mem\['memory'\]}, Score: {mem\['score'\]:.4f}")

\# 3\. Get All Memories: Retrieving a user's entire memory log  
all\_memories \= m\_self\_hosted.get\_all(user\_id=user\_id)  
print(f"\\nAll memories for user '{user\_id}':")  
\# The 'memories' key contains the list of memory objects  
if 'memories' in all\_memories and all\_memories\['memories'\]:  
    for mem in all\_memories\['memories'\]:  
        print(f"  \- {mem\['memory'\]}")  
      
    \# Get the ID of the first memory for update/history examples  
    memory\_id\_to\_update \= all\_memories\['memories'\]\['id'\]

    \# 4\. Update Memory: Modifying an existing memory  
    update\_data \= "The user now prefers the 'Sunset Ridge' trail over 'Eagle Peak'."  
    result\_update \= m\_self\_hosted.update(memory\_id=memory\_id\_to\_update, data=update\_data)  
    print(f"\\nUpdate memory result: {result\_update}")

    \# 5\. Get History: Tracking a memory's evolution  
    memory\_history \= m\_self\_hosted.history(memory\_id=memory\_id\_to\_update)  
    print(f"\\nHistory for memory ID '{memory\_id\_to\_update}':")  
    print(memory\_history)  
else:  
    print("No memories found to demonstrate update/history.")

### **4.3. Configuring Graph-based Memory in Mem0**

A key feature of Mem0 is its ability to use a graph database as a backend, which enables it to understand and traverse the relationships between memories.22 This elevates the memory system from a simple list of facts to a connected web of knowledge, mirroring the architecture of GraphRAG but for the agent's personal experiences.

When a graph store like Neo4j is configured (as shown in the initialization example), Mem0 automatically performs an additional step during the add() operation: it uses an LLM to extract entities and relationships from the input and populates the graph database. For example, when adding the memory "My friend name is john and john has a dog named tommy", Mem0 would create nodes for 'John' and 'Tommy' and a relationship like HAS\_PET between them.24

This creates a dynamic, evolving memory graph for each user. When a search is performed, Mem0 can leverage this graph structure in addition to vector similarity, allowing it to answer more complex, relational queries about the user's past, such as "Who are the people I have mentioned in the context of my hobbies?". This capability is not standalone; the graph store works in conjunction with the vector store to provide a hybrid retrieval mechanism.24

### **Table 4.1: Mem0 Python SDK API Reference**

To serve as a practical guide for implementation, the following table summarizes the core methods of the Mem0 Python SDK.

| Method | Parameters | Description | Example Usage | Return Value |
| :---- | :---- | :---- | :---- | :---- |
| Memory() | None | Initializes a default, in-memory instance of Mem0. Not persistent. | m \= Memory() | Memory object. |
| Memory.from\_config() | config: Dict | Initializes Mem0 with a specific configuration for backend stores (vector, graph, etc.).41 | m \= Memory.from\_config(config\_dict) | Memory object. |
| add() | data: Union\], user\_id: str, metadata: Optional,... | Stores a new memory. Intelligently processes text or conversation lists to extract facts.22 | m.add("I am a vegetarian", user\_id="alice") | Dict with status message. |
| search() | query: str, user\_id: str, limit: int, threshold: float,... | Performs a semantic search to find memories relevant to the query for a specific user.22 | m.search("my diet", user\_id="alice") | List of relevant memories with scores. |
| update() | memory\_id: str, data: str | Modifies the content of an existing memory identified by its unique ID.22 | m.update(mem\_id, "I am now vegan.") | Dict with status message. |
| get\_all() | user\_id: str, limit: int, page: int,... | Retrieves all stored memories for a given user, with support for pagination.22 | m.get\_all(user\_id="alice") | Dict containing a list of all memory objects. |
| history() | memory\_id: str | Returns the change log for a specific memory, showing how it has evolved over time.22 | m.history(memory\_id=mem\_id) | Dict detailing the history of changes. |

With this comprehensive understanding of the Mem0 SDK, it is now possible to build the memory-handling components of the MCP server.

## **Section 5: Assembling the MCP Server: Exposing Tools to the Agent**

This section details the final integration step: bringing the GraphRAG knowledge core (Section 3\) and the Mem0 memory layer (Section 4\) together into a single, cohesive service. This service will be an MCP server that exposes the defined toolset to an AI agent, acting as a crucial abstraction layer that hides the underlying implementation complexity. The structure will be based on the mcp-mem0 repository, which serves as an excellent template for building MCP servers with Python and FastAPI.8

The MCP server effectively becomes an **Agent Abstraction Layer**. The agent's reasoning process is complex enough without needing to manage different SDKs, CLI wrappers, or database connections. By exposing a clean, stable API contract (e.g., query\_knowledge\_base, search\_user\_memory), the MCP server allows the backend implementation to be modified or even completely swapped (e.g., switching from Microsoft's GraphRAG to the neo4j-graphrag library) without requiring any changes to the agent's logic. This promotes modularity, testability, and a clear separation of concerns, which are essential for building robust, enterprise-grade AI systems.42

### **5.1. Structuring the MCP Server with FastAPI and SSE**

The server will be built using FastAPI, a modern, high-performance Python web framework. Communication with the agent will be handled via Server-Sent Events (SSE), a lightweight protocol for pushing data from the server to the client, which is well-supported by MCP.7

**Project Structure:**

mcp\_server/  
├── main.py                 \# FastAPI app, MCP endpoint, and lifespan management  
├── knowledge\_provider.py   \# Implementation of the GraphRAG tool  
├── memory\_provider.py      \# Implementation of the Mem0 tools  
├──.env                    \# Environment variables (API keys, DB credentials)  
└── pyproject.toml          \# Project dependencies (fastapi, uvicorn, mem0ai, etc.)

**Boilerplate main.py:**

This initial main.py sets up the FastAPI application and the /sse endpoint for MCP communication. The tool logic will be added in subsequent steps.

Python

\# main.py  
import asyncio  
from fastapi import FastAPI, Request  
from fastapi.responses import StreamingResponse  
from contextlib import asynccontextmanager

\# Placeholder for our provider instances  
app\_state \= {}

@asynccontextmanager  
async def lifespan(app: FastAPI):  
    """  
    Manages the application's lifespan. Initializes resources on startup  
    and cleans them up on shutdown. This is a best practice for performance.  
    """  
    print("Server starting up...")  
    \# Initialize Knowledge and Memory providers here  
    \# from knowledge\_provider import GraphRAGKnowledgeProvider  
    \# from memory\_provider import Mem0MemoryProvider  
    \# app\_state\["knowledge\_provider"\] \= GraphRAGKnowledgeProvider(project\_root="/path/to/graphrag\_project")  
    \# app\_state\["memory\_provider"\] \= Mem0MemoryProvider()  
    print("Providers initialized.")  
    yield  
    print("Server shutting down...")  
    \# Add any cleanup logic here  
    app\_state.clear()

app \= FastAPI(lifespan=lifespan)

\# This is a simplified MCP tool dispatcher for demonstration  
async def dispatch\_tool\_call(tool\_name: str, params: dict):  
    if tool\_name \== "query\_knowledge\_base":  
        \# return await app\_state\["knowledge\_provider"\].query(\*\*params)  
        return {"result": "Knowledge query result placeholder"}  
    elif tool\_name \== "add\_interaction\_memory":  
        \# return await app\_state\["memory\_provider"\].add\_interaction(\*\*params)  
        return {"status": "Memory added placeholder"}  
    elif tool\_name \== "search\_user\_memory":  
        \# return await app\_state\["memory\_provider"\].search\_memories(\*\*params)  
        return {"memories": \["Memory search result placeholder"\]}  
    else:  
        return {"error": f"Tool '{tool\_name}' not found."}

@app.post("/sse")  
async def mcp\_sse\_endpoint(request: Request):  
    """  
    Main MCP endpoint using Server-Sent Events (SSE).  
    """  
    async def event\_stream():  
        \# A real implementation would parse the MCP request from the client  
        \# and handle a bidirectional stream. This is a simplified example.  
        try:  
            \# For demo, we assume the request body contains the tool call  
            request\_data \= await request.json()  
            tool\_name \= request\_data.get("tool\_name")  
            params \= request\_data.get("params", {})  
              
            \# Dispatch the tool call  
            result \= await dispatch\_tool\_call(tool\_name, params)  
              
            \# Send the result back as an SSE event  
            yield f"data: {json.dumps(result)}\\n\\n"  
        except Exception as e:  
            yield f"data: {json.dumps({'error': str(e)})}\\n\\n"

    return StreamingResponse(event\_stream(), media\_type="text/event-stream")

if \_\_name\_\_ \== "\_\_main\_\_":  
    import uvicorn  
    \# To run: uvicorn main:app \--reload  
    uvicorn.run(app, host="0.0.0.0", port=8080)

### **5.2. Implementing the GraphRAG Tool (query\_knowledge\_base)**

The knowledge\_provider.py module will encapsulate the logic for interacting with the chosen GraphRAG engine. This example uses the programmatic wrapper for the Microsoft graphrag library from Section 3.2.

Python

\# knowledge\_provider.py  
import subprocess  
import asyncio

class GraphRAGKnowledgeProvider:  
    def \_\_init\_\_(self, project\_root: str):  
        self.project\_root \= project\_root  
        print(f"GraphRAG Provider initialized for root: {self.project\_root}")

    async def query(self, query: str, search\_type: str, user\_context: str \= None) \-\> dict:  
        """  
        Asynchronously queries the GraphRAG index.  
        """  
        if search\_type not in \["global", "local"\]:  
            return {"error": "Invalid search\_type. Must be 'global' or 'local'."}

        \# Prepend user context to the query if provided  
        if user\_context:  
            full\_query \= f"Based on the context that '{user\_context}', answer the following: {query}"  
        else:  
            full\_query \= query  
          
        command \= \[  
            "python", "-m", "graphrag.query",  
            "--root", self.project\_root,  
            "--method", search\_type,  
            full\_query  
        \]

        process \= await asyncio.create\_subprocess\_exec(  
            \*command,  
            stdout=asyncio.subprocess.PIPE,  
            stderr=asyncio.subprocess.PIPE  
        )  
          
        stdout, stderr \= await process.communicate()

        if process.returncode\!= 0:  
            return {"error": stderr.decode()}  
          
        \# Parse the output to extract the answer  
        response\_text \= stdout.decode()  
        try:  
            answer \= response\_text.split("Response:").strip()  
        except IndexError:  
            answer \= "Could not parse response from GraphRAG."

        return {"answer": answer}

### **5.3. Implementing the Mem0 Tools**

The memory\_provider.py module will contain the class that interacts with the Mem0 SDK. This example assumes a self-hosted Mem0 instance is configured as shown in Section 4.2.

Python

\# memory\_provider.py  
from mem0 import Memory  
from typing import List, Dict, Optional

class Mem0MemoryProvider:  
    def \_\_init\_\_(self):  
        \# Initialize Mem0 from a configuration dictionary  
        config \= {  
            "vector\_store": {"provider": "qdrant", "config": {"host": "localhost", "port": 6333}},  
            "graph\_store": {"provider": "neo4j", "config": {"url": "bolt://localhost:7687", "username": "neo4j", "password": "password"}}  
        }  
        self.mem0\_instance \= Memory.from\_config(config)  
        print("Mem0 Provider initialized.")

    async def add\_interaction(self, user\_id: str, conversation\_turn: List, metadata: Optional \= None) \-\> Dict:  
        """Adds a conversation turn to the user's memory."""  
        try:  
            result \= self.mem0\_instance.add(data=conversation\_turn, user\_id=user\_id, metadata=metadata)  
            return result  
        except Exception as e:  
            return {"error": str(e)}

    async def search\_memories(self, user\_id: str, query: str, limit: int \= 5) \-\> Dict:  
        """Searches a user's memories."""  
        try:  
            results \= self.mem0\_instance.search(query=query, user\_id=user\_id, limit=limit)  
            return {"memories": results}  
        except Exception as e:  
            return {"error": str(e)}

### **5.4. Final Assembly with Lifespan Management**

The final step is to update main.py to use these provider classes and properly manage their lifecycle. Using FastAPI's lifespan context manager is crucial for production readiness, as it ensures that expensive resources like database connections and model initializations happen only once at server startup, not on every request.8

**Final main.py:**

Python

\# main.py  
import asyncio  
import json  
from fastapi import FastAPI, Request  
from fastapi.responses import StreamingResponse  
from contextlib import asynccontextmanager

from knowledge\_provider import GraphRAGKnowledgeProvider  
from memory\_provider import Mem0MemoryProvider

app\_state \= {}

@asynccontextmanager  
async def lifespan(app: FastAPI):  
    """  
    Initializes providers on startup and stores them in the app state.  
    """  
    print("Server starting up...")  
    app\_state\["knowledge\_provider"\] \= GraphRAGKnowledgeProvider(project\_root="./graphrag\_project")  
    app\_state\["memory\_provider"\] \= Mem0MemoryProvider()  
    print("Providers initialized and ready.")  
    yield  
    print("Server shutting down...")  
    app\_state.clear()

app \= FastAPI(lifespan=lifespan)

async def dispatch\_tool\_call(tool\_name: str, params: dict):  
    """Routes incoming tool calls to the correct provider method."""  
    provider \= None  
    method\_to\_call \= None

    if tool\_name \== "query\_knowledge\_base":  
        provider \= app\_state.get("knowledge\_provider")  
        method\_to\_call \= provider.query if provider else None  
    elif tool\_name \== "add\_interaction\_memory":  
        provider \= app\_state.get("memory\_provider")  
        method\_to\_call \= provider.add\_interaction if provider else None  
    elif tool\_name \== "search\_user\_memory":  
        provider \= app\_state.get("memory\_provider")  
        method\_to\_call \= provider.search\_memories if provider else None  
      
    if method\_to\_call:  
        return await method\_to\_call(\*\*params)  
    else:  
        return {"error": f"Tool '{tool\_name}' or its provider is not available."}

@app.post("/sse")  
async def mcp\_sse\_endpoint(request: Request):  
    """Main MCP endpoint using Server-Sent Events (SSE)."""  
    async def event\_stream():  
        try:  
            request\_data \= await request.json()  
            tool\_name \= request\_data.get("tool\_name")  
            params \= request\_data.get("params", {})  
              
            result \= await dispatch\_tool\_call(tool\_name, params)  
              
            yield f"data: {json.dumps(result)}\\n\\n"  
        except Exception as e:  
            yield f"data: {json.dumps({'error': str(e)})}\\n\\n"

    return StreamingResponse(event\_stream(), media\_type="text/event-stream")

\# To run this server:  
\# 1\. Ensure your graphrag\_project is set up and indexed.  
\# 2\. Ensure your Mem0 backend (Qdrant, Neo4j) is running via Docker.  
\# 3\. Run: uvicorn main:app \--host 0.0.0.0 \--port 8080

This assembled server provides a robust, scalable, and modular foundation for an advanced AI agent, exposing the powerful combined capabilities of GraphRAG and Mem0 through a standardized protocol.

## **Section 6: Advanced Strategies and Best Practices**

Moving from a functional prototype to a robust, intelligent, and optimized production system requires a focus on advanced strategies. This section provides expert guidance on agentic orchestration, creating feedback loops, performance optimization, and scaling the integrated GraphRAG and Mem0 system.

### **6.1. Agentic Orchestration: When to Query Knowledge vs. Access Memory**

The core intelligence of the agent lies in its ability to decide *which* tool to use and *how* to use it. This decision-making process can be guided through careful prompt engineering and more advanced routing mechanisms.

#### **Prompt Engineering for Tool Selection**

The most direct way to guide the agent is through its system prompt. The prompt should explicitly instruct the agent on the purpose of each tool and provide clear criteria for their use. This makes the agent's reasoning process more reliable and predictable.43

**Example System Prompt Fragment:**

You are an expert research assistant with access to two primary information sources: a deep knowledge base and a personal memory of our conversations. Use your tools as follows:

1. **search\_user\_memory(query)**: Use this tool FIRST to understand the user's context, preferences, or recall past interactions. For example, if the user asks "What do you know about me?" or refers to a previous topic, use this tool.  
2. **query\_knowledge\_base(query, search\_type, user\_context)**: Use this tool to answer general domain questions or find specific facts within the knowledge base.  
   * Set search\_type='global' for broad, thematic questions like "What are the main themes?".  
   * Set search\_type='local' for specific questions about entities like "Tell me about Project X."  
   * Optionally, use the output from search\_user\_memory to populate the user\_context parameter to get more personalized knowledge results.  
3. **add\_interaction\_memory(conversation\_turn)**: After every meaningful exchange where new information is learned about the user or a topic, use this tool to save the context for future reference.

#### **Agentic Routers**

For more complex scenarios, a dedicated "router" model can be employed. This is an advanced RAG pattern where an initial, smaller LLM call is made to classify the user's query and determine the best execution path.42 The router can decide which tool to call, what

search\_type to use, or even if multiple tool calls are needed to satisfy the query. This adds a layer of intelligent dispatching that can improve both accuracy and efficiency.

### **6.2. Creating a Feedback Loop: Evolving Memory with Knowledge and Vice Versa**

The true power of this integrated architecture is unlocked by creating a feedback loop between the knowledge base and the memory layer. This transforms the system from a static repository with a dynamic memory into a self-improving ecosystem.

* **From Knowledge to Memory:** This is the most direct feedback path. When the agent retrieves a crucial piece of information from the GraphRAG knowledge base and successfully uses it to answer a user's question, this entire interaction (query, retrieved context, and final answer) should be stored in Mem0 using the add\_interaction\_memory tool. This creates a new memory, such as, "I taught the user about 'dynamic community selection' using information from the knowledge base." The next time the user asks about this topic, the agent can recall this memory and provide a more nuanced or follow-up response, demonstrating true learning.  
* **From Memory to Knowledge:** This is a more advanced, system-level feedback loop. By analyzing the memory data aggregated across many users (anonymously, of course), system administrators can identify gaps in the knowledge base. For example, if logs show that dozens of users are asking questions about "Project Titan" and the query\_knowledge\_base tool consistently returns poor or empty results, this is a strong signal that the document corpus lacks information on this topic. This insight can trigger a workflow to find and index new documents about "Project Titan," thereby evolving and improving the core GraphRAG knowledge base in direct response to user needs. This turns user interaction data into a valuable asset for knowledge base curation.

### **6.3. Performance Optimization: Caching, Indexing, and Prompting**

Latency and cost are critical considerations in production AI systems. Several optimization techniques can be applied to the GraphRAG and Mem0 server.

* **Caching Strategies:** Implement a multi-level caching strategy. Cache the final responses to common, identical queries to the GraphRAG engine. For Mem0, cache the results of frequent search\_user\_memory calls, especially within a single session. This can dramatically reduce redundant LLM calls and database lookups, lowering both latency and operational costs.43 Cache invalidation can be managed based on timestamps or when the underlying data sources are updated.  
* **Vector Database Tuning:** The performance of both GraphRAG and Mem0 heavily relies on the efficiency of their underlying vector stores.  
  * **Embedding Models:** Experiment with different embedding models. While large models may offer higher semantic richness, smaller, domain-specific models can sometimes provide better retrieval precision for a specific use case at a fraction of the computational cost.42  
  * **Tuning top\_k:** The number of documents retrieved (top\_k) is a critical parameter. A smaller k (e.g., 3-5) results in faster retrieval and a smaller context for the LLM, reducing cost and latency. A larger k may increase recall but also introduces more noise. The optimal value should be determined empirically based on the specific use case.43  
  * **Metadata Filtering:** Leverage metadata filters in the vector database whenever possible. If a query contains a date or a specific category, pre-filtering the search space can drastically improve speed and relevance.42  
* **Prompt Formatting:** The way retrieved context is presented to the final generation LLM significantly impacts the quality of the response. Clearly delineate the different sources of information within the prompt using markdown or special tokens. This prevents the LLM from confusing retrieved knowledge with the user's direct query or its own instructions.43  
  **Example Prompt Structure:**  
  You are a helpful AI assistant. Answer the user's question based on the provided knowledge and memory context.

  \#\#\# Retrieved Knowledge from GraphRAG:  
  {graphrag\_context}

  \#\#\# Relevant Memories from Past Interactions:  
  {mem0\_context}

  \#\#\# User's Question:  
  {user\_query}

  \#\#\# Answer:

### **6.4. Scalability and Multi-User Deployment Considerations**

The proposed architecture is designed with scalability and multi-user support in mind.

* **Multi-Tenancy:** The system is inherently ready for multiple users because all Mem0 API calls are keyed by a user\_id.22 This ensures that each user's memory is isolated and secure. The GraphRAG knowledge base is typically shared among all users, acting as a common source of truth.  
* **Stateless Server, Stateful Backend:** The MCP server itself, built with FastAPI, is stateless. This means it can be containerized and horizontally scaled using orchestrators like Kubernetes. All state is managed by the backend databases (Neo4j, Qdrant, etc.). This is a standard, robust pattern for building scalable cloud applications.  
* **Production-Grade Infrastructure:** For production deployment, it is highly recommended to use managed, production-grade services for the backend databases. Services like Neo4j AuraDB, Qdrant Cloud, or Amazon Neptune Analytics provide the reliability, scalability, and maintenance required to handle the load of a large-scale, multi-user system.24

By implementing these advanced strategies, a developer can elevate the integrated GraphRAG and Mem0 server from a powerful proof-of-concept to a truly intelligent, efficient, and scalable system capable of supporting a new class of expert AI agents.

### **Conclusion**

The architecture presented in this report, combining GraphRAG's structured knowledge retrieval with Mem0's agentic memory layer under the unifying framework of MCP, represents a significant step forward in the design of intelligent AI systems. It moves beyond the limitations of simple, stateless RAG by creating an agent that possesses both deep, relational domain expertise and a personalized, evolving memory of its interactions.

The key takeaways from this analysis are:

1. **Complementary Strengths:** GraphRAG provides the "what"—a structured, queryable knowledge base of a domain. Mem0 provides the "who" and "when"—the stateful, personal context of user interactions. Their combination enables a level of contextual understanding that neither can achieve alone.  
2. **Architectural Soundness:** The use of MCP as an abstraction layer promotes modularity, scalability, and maintainability. It decouples the agent's reasoning from the complex implementation of its tools, allowing for independent development and evolution of system components.  
3. **Implementation Flexibility:** There are multiple viable paths to implementing the GraphRAG core, from the reference Microsoft library to the SDK-native neo4j-graphrag and the modular LlamaIndex framework. The optimal choice depends on specific project requirements regarding control, existing infrastructure, and development velocity.  
4. **The Importance of Agentic Orchestration:** The system's ultimate intelligence is not just in its components but in the agent's ability to strategically choose when to access deep knowledge versus personal memory. This orchestration, guided by careful prompt engineering and routing logic, is what unlocks the system's full potential.  
5. **Path to Self-Improvement:** The proposed feedback loops, where knowledge informs memory and aggregated memory patterns inform knowledge base curation, create a pathway for the system to learn and improve over time, becoming more valuable with every interaction.

By following the principles and practical guides laid out in this report, engineers and architects can build sophisticated AI systems that are not merely information retrievers but true digital partners—capable of remembering, reasoning, and collaborating on complex tasks with a rich, multi-layered understanding of both their domain and their users.

#### **Works cited**

1. What Is RAG Architecture? A New Approach to LLMs \- Cohere, accessed July 6, 2025, [https://cohere.com/blog/rag-architecture](https://cohere.com/blog/rag-architecture)  
2. What Is Retrieval Augmented Generation (RAG)? \- Pure Storage, accessed July 6, 2025, [https://www.purestorage.com/knowledge/what-is-retrieval-augmented-generation.html](https://www.purestorage.com/knowledge/what-is-retrieval-augmented-generation.html)  
3. GraphRAG: Unlocking LLM discovery on narrative private data \- Microsoft Research, accessed July 6, 2025, [https://www.microsoft.com/en-us/research/blog/graphrag-unlocking-llm-discovery-on-narrative-private-data/](https://www.microsoft.com/en-us/research/blog/graphrag-unlocking-llm-discovery-on-narrative-private-data/)  
4. Welcome \- GraphRAG, accessed July 6, 2025, [https://microsoft.github.io/graphrag/](https://microsoft.github.io/graphrag/)  
5. What is Mem0? \- Mem0, accessed July 6, 2025, [https://docs.mem0.ai/what-is-mem0](https://docs.mem0.ai/what-is-mem0)  
6. 5 Advanced RAG Architectures Beyond Traditional Methods \- MachineLearningMastery.com, accessed July 6, 2025, [https://machinelearningmastery.com/5-advanced-rag-architectures-beyond-traditional-methods/](https://machinelearningmastery.com/5-advanced-rag-architectures-beyond-traditional-methods/)  
7. mem0ai/mem0-mcp \- GitHub, accessed July 6, 2025, [https://github.com/mem0ai/mem0-mcp](https://github.com/mem0ai/mem0-mcp)  
8. coleam00/mcp-mem0: MCP server for long term agent memory with Mem0. Also useful as a template to get you started building your own MCP server with Python\! \- GitHub, accessed July 6, 2025, [https://github.com/coleam00/mcp-mem0](https://github.com/coleam00/mcp-mem0)  
9. microsoft/graphrag: A modular graph-based Retrieval-Augmented Generation (RAG) system, accessed July 6, 2025, [https://github.com/microsoft/graphrag](https://github.com/microsoft/graphrag)  
10. What is GraphRAG? \- IBM, accessed July 6, 2025, [https://www.ibm.com/think/topics/graphrag](https://www.ibm.com/think/topics/graphrag)  
11. GraphRAG: A Complete Guide from Concept to Implementation \- Analytics Vidhya, accessed July 6, 2025, [https://www.analyticsvidhya.com/blog/2024/11/graphrag/](https://www.analyticsvidhya.com/blog/2024/11/graphrag/)  
12. Overview \- GraphRAG, accessed July 6, 2025, [https://microsoft.github.io/graphrag/index/overview/](https://microsoft.github.io/graphrag/index/overview/)  
13. Inputs \- GraphRAG \- Microsoft Open Source, accessed July 6, 2025, [https://microsoft.github.io/graphrag/index/inputs/](https://microsoft.github.io/graphrag/index/inputs/)  
14. GraphRAG: The Practical Guide for Cost-Effective Document Analysis with Knowledge Graphs \- LearnOpenCV, accessed July 6, 2025, [https://learnopencv.com/graphrag-explained-knowledge-graphs-medical/](https://learnopencv.com/graphrag-explained-knowledge-graphs-medical/)  
15. GraphRAG Implementation with LlamaIndex, accessed July 6, 2025, [https://docs.llamaindex.ai/en/stable/examples/cookbooks/GraphRAG\_v1/](https://docs.llamaindex.ai/en/stable/examples/cookbooks/GraphRAG_v1/)  
16. GraphRAG \- Read the Docs, accessed July 6, 2025, [http://langchain-graphrag.readthedocs.io/](http://langchain-graphrag.readthedocs.io/)  
17. GraphRAG: Improving global search via dynamic community selection \- Microsoft Research, accessed July 6, 2025, [https://www.microsoft.com/en-us/research/blog/graphrag-improving-global-search-via-dynamic-community-selection/](https://www.microsoft.com/en-us/research/blog/graphrag-improving-global-search-via-dynamic-community-selection/)  
18. Microsoft Research Blog \- GraphRAG, accessed July 6, 2025, [https://microsoft.github.io/graphrag/blog\_posts/](https://microsoft.github.io/graphrag/blog_posts/)  
19. Overview \- Mem0, accessed July 6, 2025, [https://docs.mem0.ai/overview](https://docs.mem0.ai/overview)  
20. Mem0 \- The Memory Layer for your AI Apps, accessed July 6, 2025, [https://mem0.ai/](https://mem0.ai/)  
21. Agent Memory \- How should it work? : r/Rag \- Reddit, accessed July 6, 2025, [https://www.reddit.com/r/Rag/comments/1l9zudk/agent\_memory\_how\_should\_it\_work/](https://www.reddit.com/r/Rag/comments/1l9zudk/agent_memory_how_should_it_work/)  
22. into-the-night/mem0-feat-gemini: The Memory layer for your AI apps (now with Gemini\!), accessed July 6, 2025, [https://github.com/into-the-night/mem0-feat-gemini](https://github.com/into-the-night/mem0-feat-gemini)  
23. markmbain/mem0ai-mem0: The memory layer for Personalized AI \- GitHub, accessed July 6, 2025, [https://github.com/markmbain/mem0ai-mem0](https://github.com/markmbain/mem0ai-mem0)  
24. Overview \- Mem0, accessed July 6, 2025, [https://docs.mem0.ai/open-source/graph\_memory/overview](https://docs.mem0.ai/open-source/graph_memory/overview)  
25. Your First GraphRAG Demo \- A Video Walkthrough \- Microsoft Community Hub, accessed July 6, 2025, [https://techcommunity.microsoft.com/blog/aiplatformblog/your-first-graphrag-demo---a-video-walkthrough/4410246](https://techcommunity.microsoft.com/blog/aiplatformblog/your-first-graphrag-demo---a-video-walkthrough/4410246)  
26. Integrations \- Memgraph, accessed July 6, 2025, [https://memgraph.com/docs/ai-ecosystem/integrations](https://memgraph.com/docs/ai-ecosystem/integrations)  
27. Your 2025 Roadmap to Becoming an AI Engineer | by Rudresh Narwal ‍ \- Medium, accessed July 6, 2025, [https://medium.com/@rudresh.narwal/your-2025-roadmap-to-becoming-an-ai-engineer-4b3b581e059f](https://medium.com/@rudresh.narwal/your-2025-roadmap-to-becoming-an-ai-engineer-4b3b581e059f)  
28. mem0ai/mem0: Memory for AI Agents; Announcing ... \- GitHub, accessed July 6, 2025, [https://github.com/mem0ai/mem0](https://github.com/mem0ai/mem0)  
29. neo4j-graphrag-python documentation, accessed July 6, 2025, [https://neo4j.com/docs/neo4j-graphrag-python/current/](https://neo4j.com/docs/neo4j-graphrag-python/current/)  
30. Getting Started \- GraphRAG \- Microsoft Open Source, accessed July 6, 2025, [https://microsoft.github.io/graphrag/get\_started/](https://microsoft.github.io/graphrag/get_started/)  
31. GraphRAG Python Package: Accelerating GenAI With Knowledge Graphs \- Neo4j, accessed July 6, 2025, [https://neo4j.com/blog/news/graphrag-python-package/](https://neo4j.com/blog/news/graphrag-python-package/)  
32. Graph RAG \- ️ LangChain, accessed July 6, 2025, [https://python.langchain.com/docs/integrations/retrievers/graph\_rag/](https://python.langchain.com/docs/integrations/retrievers/graph_rag/)  
33. How to Add HTTP API for GraphRAG? \- Aident AI, accessed July 6, 2025, [https://aident.ai/blog/how-to-add-http-api-for-graphrag](https://aident.ai/blog/how-to-add-http-api-for-graphrag)  
34. Simple python app that reproduce default indexing · microsoft graphrag · Discussion \#694, accessed July 6, 2025, [https://github.com/microsoft/graphrag/discussions/694](https://github.com/microsoft/graphrag/discussions/694)  
35. User Guide: RAG — neo4j-graphrag-python documentation, accessed July 6, 2025, [https://neo4j.com/docs/neo4j-graphrag-python/current/user\_guide\_rag.html](https://neo4j.com/docs/neo4j-graphrag-python/current/user_guide_rag.html)  
36. API Documentation — neo4j-graphrag-python documentation, accessed July 6, 2025, [https://neo4j.com/docs/neo4j-graphrag-python/current/api.html](https://neo4j.com/docs/neo4j-graphrag-python/current/api.html)  
37. graphrag \- PyPI, accessed July 6, 2025, [https://pypi.org/project/graphrag/](https://pypi.org/project/graphrag/)  
38. Neo4j GraphRAG for Python \- GitHub, accessed July 6, 2025, [https://github.com/neo4j/neo4j-graphrag-python](https://github.com/neo4j/neo4j-graphrag-python)  
39. Quickstart \- Mem0, accessed July 6, 2025, [https://docs.mem0.ai/openmemory/quickstart](https://docs.mem0.ai/openmemory/quickstart)  
40. Quickstart \- Mem0, accessed July 6, 2025, [https://docs.mem0.ai/quickstart](https://docs.mem0.ai/quickstart)  
41. Python SDK Quickstart \- Mem0, accessed July 6, 2025, [https://docs.mem0.ai/open-source/python-quickstart](https://docs.mem0.ai/open-source/python-quickstart)  
42. Chapter 5: Best Practices for RAG | by Marc Haraoui | Jun, 2025 | Medium, accessed July 6, 2025, [https://medium.com/@marcharaoui/chapter-5-best-practices-for-rag-7770fce8ac81](https://medium.com/@marcharaoui/chapter-5-best-practices-for-rag-7770fce8ac81)  
43. Integrating Agentic RAG with MCP Servers: Technical ... \- Omar Santos, accessed July 6, 2025, [https://becomingahacker.org/integrating-agentic-rag-with-mcp-servers-technical-implementation-guide-1aba8fd4e442](https://becomingahacker.org/integrating-agentic-rag-with-mcp-servers-technical-implementation-guide-1aba8fd4e442)  
44. Understanding Modern RAG Architectures: From Simple to Complex \- Medium, accessed July 6, 2025, [https://medium.com/@tj.ruesch/understanding-modern-rag-architectures-from-simple-to-complex-6eef17f702ba](https://medium.com/@tj.ruesch/understanding-modern-rag-architectures-from-simple-to-complex-6eef17f702ba)  
45. Memory and RAG — AutoGen \- Microsoft Open Source, accessed July 6, 2025, [https://microsoft.github.io/autogen/stable//user-guide/agentchat-user-guide/memory.html](https://microsoft.github.io/autogen/stable//user-guide/agentchat-user-guide/memory.html)  
46. Seeking Advice on Memory Management for Multi-User LLM Agent System \- Reddit, accessed July 6, 2025, [https://www.reddit.com/r/AI\_Agents/comments/1jhub84/seeking\_advice\_on\_memory\_management\_for\_multiuser/](https://www.reddit.com/r/AI_Agents/comments/1jhub84/seeking_advice_on_memory_management_for_multiuser/)