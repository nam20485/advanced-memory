"""
Knowledge provider using Neo4j GraphRAG for structured knowledge retrieval.
"""

import asyncio
import logging
import time
from typing import Any, Dict, List, Optional

from neo4j import GraphDatabase, Driver
from neo4j_graphrag.llm import OpenAILLM
from neo4j_graphrag.embeddings import OpenAIEmbeddings
from neo4j_graphrag.retrievers import VectorCypherRetriever
from neo4j_graphrag.generation import GraphRAG
from neo4j_graphrag.pipeline import SimpleKGPipeline

from ..config import settings
from ..models.knowledge_models import (
    KnowledgeQuery, KnowledgeResult, SearchType, Entity, 
    Relationship, Community, EntityType, RelationshipType
)

logger = logging.getLogger(__name__)


class GraphRAGKnowledgeProvider:
    """
    Provides GraphRAG-based knowledge retrieval using Neo4j.
    
    This provider implements both global and local search capabilities,
    leveraging Neo4j's graph database for storing and querying the
    knowledge graph built from source documents.
    """
    
    def __init__(self):
        """Initialize the GraphRAG knowledge provider."""
        self.driver: Optional[Driver] = None
        self.llm: Optional[OpenAILLM] = None
        self.embedder: Optional[OpenAIEmbeddings] = None
        self.graphrag: Optional[GraphRAG] = None
        self._initialized = False
    
    async def initialize(self) -> None:
        """Initialize the Neo4j connection and GraphRAG components."""
        try:
            # Initialize Neo4j driver
            self.driver = GraphDatabase.driver(
                settings.neo4j_uri,
                auth=(settings.neo4j_username, settings.neo4j_password)
            )
            
            # Test connection
            await asyncio.get_event_loop().run_in_executor(
                None, self._test_connection
            )
            
            # Initialize LLM and embedder
            self.llm = OpenAILLM(
                model_name=settings.openai_model,
                api_key=settings.openai_api_key
            )
            
            self.embedder = OpenAIEmbeddings(
                model_name=settings.openai_embedding_model,
                api_key=settings.openai_api_key
            )
            
            # Initialize GraphRAG system
            retriever = VectorCypherRetriever(
                driver=self.driver,
                index_name=settings.vector_index_name,
                embedder=self.embedder,
                # Custom Cypher query for enhanced retrieval
                retrieval_query="""
                MATCH (node:Chunk {embedding: $embedding})
                CALL db.index.vector.queryNodes($index_name, $k, $embedding) 
                YIELD node as chunk, score
                MATCH (chunk)-[:MENTIONS]->(entity:Entity)
                OPTIONAL MATCH (entity)-[rel:RELATED_TO]-(connected:Entity)
                RETURN chunk.text as text, 
                       collect(DISTINCT entity.name) as entities,
                       collect(DISTINCT {source: entity.name, target: connected.name, type: type(rel)}) as relationships,
                       score
                ORDER BY score DESC
                """
            )
            
            self.graphrag = GraphRAG(
                retriever=retriever,
                llm=self.llm
            )
            
            self._initialized = True
            logger.info("GraphRAG knowledge provider initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize GraphRAG provider: {e}")
            raise
    
    def _test_connection(self) -> None:
        """Test the Neo4j connection."""
        with self.driver.session() as session:
            result = session.run("RETURN 1 as test")
            assert result.single()["test"] == 1
    
    async def index_documents(self, documents: List[Dict[str, Any]]) -> bool:
        """
        Index documents into the knowledge graph.
        
        Args:
            documents: List of documents to index, each with 'content' and 'metadata'
            
        Returns:
            bool: True if indexing was successful
        """
        if not self._initialized:
            await self.initialize()
        
        try:
            # Create indexing pipeline
            pipeline = SimpleKGPipeline(
                llm=self.llm,
                embedder=self.embedder,
                driver=self.driver,
                database=settings.neo4j_database
            )
            
            # Process documents
            for doc in documents:
                await asyncio.get_event_loop().run_in_executor(
                    None, pipeline.run, doc['content']
                )
            
            logger.info(f"Successfully indexed {len(documents)} documents")
            return True
            
        except Exception as e:
            logger.error(f"Failed to index documents: {e}")
            return False
    
    async def query_knowledge_base(
        self, 
        query: str, 
        search_type: SearchType = SearchType.LOCAL,
        user_context: Optional[str] = None
    ) -> KnowledgeResult:
        """
        Query the knowledge base using GraphRAG.
        
        Args:
            query: The question to answer
            search_type: Type of search (global or local)
            user_context: Optional user context for personalization
            
        Returns:
            KnowledgeResult: The search result
        """
        if not self._initialized:
            await self.initialize()
        
        start_time = time.time()
        
        try:
            # Create knowledge query object
            knowledge_query = KnowledgeQuery(
                query=query,
                search_type=search_type,
                user_context=user_context
            )
            
            # Enhance query with user context if provided
            enhanced_query = query
            if user_context:
                enhanced_query = f"Context: {user_context}\n\nQuestion: {query}"
            
            # Execute the search
            if search_type == SearchType.GLOBAL:
                answer = await self._global_search(enhanced_query)
            else:
                answer = await self._local_search(enhanced_query)
            
            # Calculate processing time
            processing_time = int((time.time() - start_time) * 1000)
            
            # Create and return result
            return KnowledgeResult(
                query=knowledge_query,
                answer=answer,
                confidence=0.9,  # TODO: Implement confidence scoring
                processing_time_ms=processing_time
            )
            
        except Exception as e:
            logger.error(f"Failed to query knowledge base: {e}")
            # Return error result
            return KnowledgeResult(
                query=KnowledgeQuery(query=query, search_type=search_type),
                answer=f"Error processing query: {str(e)}",
                confidence=0.0,
                processing_time_ms=int((time.time() - start_time) * 1000)
            )
    
    async def _global_search(self, query: str) -> str:
        """
        Perform global search using community summaries.
        
        Args:
            query: The search query
            
        Returns:
            str: The answer from global search
        """
        # For global search, we query community summaries
        cypher_query = """
        MATCH (c:Community)
        WHERE c.level = $level
        RETURN c.title, c.summary, c.size
        ORDER BY c.size DESC
        LIMIT 20
        """
        
        with self.driver.session() as session:
            result = session.run(cypher_query, level=settings.graphrag_community_level)
            communities = [record.data() for record in result]
        
        # Use LLM to synthesize answer from community summaries
        context = "\n".join([
            f"Community: {c['title']}\nSummary: {c['summary']}\n"
            for c in communities
        ])
        
        prompt = f"""
        Based on the following community summaries, answer this question: {query}
        
        Community Information:
        {context}
        
        Please provide a comprehensive answer that synthesizes information from multiple communities.
        """
        
        return await asyncio.get_event_loop().run_in_executor(
            None, self.llm.invoke, prompt
        )
    
    async def _local_search(self, query: str) -> str:
        """
        Perform local search using GraphRAG retriever.
        
        Args:
            query: The search query
            
        Returns:
            str: The answer from local search
        """
        # Use the GraphRAG system for local search
        response = await asyncio.get_event_loop().run_in_executor(
            None, self.graphrag.search, query
        )
        
        return response.answer if hasattr(response, 'answer') else str(response)
    
    async def get_entities(self, entity_type: Optional[EntityType] = None) -> List[Entity]:
        """
        Get entities from the knowledge graph.
        
        Args:
            entity_type: Optional filter by entity type
            
        Returns:
            List[Entity]: List of entities
        """
        if not self._initialized:
            await self.initialize()
        
        cypher_query = """
        MATCH (e:Entity)
        """ + (f"WHERE e.type = '{entity_type.value}'" if entity_type else "") + """
        RETURN e.id, e.name, e.type, e.description
        LIMIT 100
        """
        
        with self.driver.session() as session:
            result = session.run(cypher_query)
            entities = []
            
            for record in result:
                entity = Entity(
                    id=record["e.id"],
                    name=record["e.name"],
                    entity_type=EntityType(record["e.type"]),
                    description=record["e.description"]
                )
                entities.append(entity)
        
        return entities
    
    async def get_relationships(self, source_id: Optional[str] = None) -> List[Relationship]:
        """
        Get relationships from the knowledge graph.
        
        Args:
            source_id: Optional filter by source entity ID
            
        Returns:
            List[Relationship]: List of relationships
        """
        if not self._initialized:
            await self.initialize()
        
        cypher_query = """
        MATCH (source:Entity)-[r:RELATED_TO]->(target:Entity)
        """ + (f"WHERE source.id = '{source_id}'" if source_id else "") + """
        RETURN source.id, target.id, type(r), r.description, r.weight
        LIMIT 100
        """
        
        with self.driver.session() as session:
            result = session.run(cypher_query)
            relationships = []
            
            for record in result:
                relationship = Relationship(
                    id=f"{record['source.id']}-{record['target.id']}",
                    source_id=record["source.id"],
                    target_id=record["target.id"],
                    relationship_type=RelationshipType.RELATED_TO,
                    description=record["r.description"],
                    weight=record["r.weight"]
                )
                relationships.append(relationship)
        
        return relationships
    
    async def close(self) -> None:
        """Close the Neo4j connection."""
        if self.driver:
            self.driver.close()
            logger.info("Neo4j connection closed")
