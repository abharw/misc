import asyncio
import json
import logging
import os
from typing import Any, Dict, List, Optional, Union

import websockets
from websockets.exceptions import ConnectionClosed
from livekit.agents.stt.stt import (
    STT, 
    STTCapabilities, 
    SpeechEvent, 
    SpeechEventType,
    SpeechData,
    RecognizeStream
)
from livekit.agents.types import NOT_GIVEN, APIConnectOptions, NotGivenOr
from livekit.rtc.audio_frame import AudioFrame

logger = logging.getLogger(__name__)


class SonioxSTT(STT):
    """Soniox STT integration for LiveKit Agents."""
    
    WEBSOCKET_URL = "wss://stt-rt.soniox.com/transcribe-websocket"
     
    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        model: str = "stt-rt-preview",
        language: str = "tr",
        sample_rate: int = 16000,
        interim_results: bool = True,
        punctuate: bool = True,
        diarize: bool = False,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        timeout: float = 30.0,
    ) -> None:
        """
        Initialize Soniox STT.
        
        Args:
            api_key: Soniox API key. If not provided, will look for SONIOX_API_KEY env var.
            model: Soniox model to use (stt-rt-preview, etc.)
            language: Language code or 'auto' for automatic detection
            sample_rate: Audio sample rate in Hz
            interim_results: Whether to return interim results
            punctuate: Whether to add punctuation
            diarize: Whether to perform speaker diarization
            max_retries: Maximum number of retry attempts
            retry_delay: Delay between retries in seconds
            timeout: Request timeout in seconds
        """
        self.api_key = api_key or os.getenv("SONIOX_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Soniox API key is required. Set SONIOX_API_KEY environment variable "
                "or pass api_key parameter."
            )
        
        self.model = model
        self.language = language
        self.sample_rate = sample_rate
        self.interim_results = interim_results
        self.punctuate = punctuate
        self.diarize = diarize
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.timeout = timeout
        
        capabilities = STTCapabilities(
            streaming=True,
            interim_results=interim_results
        )
        
        super().__init__(capabilities=capabilities)
        
        logger.info(
            f"Soniox STT initialized with model={model}, language={language}, "
            f"sample_rate={sample_rate}"
        )
    
    @property
    def label(self) -> str:
        """Return a human-readable label for this STT provider."""
        return f"Soniox ({self.model})"
    

    
    async def _recognize_impl(
        self, 
        buffer: Union[List[AudioFrame], AudioFrame], 
        *, 
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions = APIConnectOptions()
    ) -> SpeechEvent:
        """
        Recognize speech from audio buffer.
        
        Note: For real-time streaming, use the stream() method instead.
        This method is for single-shot recognition and creates a temporary connection.
        
        Args:
            buffer: Audio buffer containing speech data
            language: Language code (overrides instance language)
            
        Returns:
            SpeechEvent with transcription results
        """
        stream = self.stream(language=language)
        
        try:
            # Send audio data - handle both single frame and list of frames
            if isinstance(buffer, list):
                for frame in buffer:
                    await stream.write(frame)
            else:
                await stream.write(buffer)
            
            # Flush to get final results
            stream.flush()
            
            # Wait for final result with timeout
            timeout = 5.0  
            start_time = asyncio.get_event_loop().time()
            
            async for event in stream:
                if event.type == SpeechEventType.FINAL_TRANSCRIPT:
                    return event
                
                if asyncio.get_event_loop().time() - start_time > timeout:
                    break
            
            # Return empty result if no transcription received
            return SpeechEvent(
                type=SpeechEventType.FINAL_TRANSCRIPT,
                alternatives=[
                    SpeechData(
                        language=language or self.language if language != NOT_GIVEN else self.language,
                        text=""
                    )
                ]
            )
            
        finally:
            await stream.aclose()
    
    def stream(
        self, 
        *, 
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions = APIConnectOptions()
    ) -> "SonioxRecognizeStream":
        """
        Create a streaming speech recognition session.
        
        Args:
            language: Language code (overrides instance language)
            conn_options: Connection options
            
        Returns:
            SonioxRecognizeStream for real-time transcription
        """
        if not self.api_key:
            raise ValueError("API key is required")
        
        actual_language = language if language != NOT_GIVEN else self.language
        if actual_language == NOT_GIVEN or actual_language is None:
            actual_language = "auto"
        
        actual_language = str(actual_language) if actual_language != NOT_GIVEN else "auto"
        
        return SonioxRecognizeStream(
            api_key=self.api_key,
            model=self.model,
            language=actual_language,
            sample_rate=self.sample_rate,
            interim_results=self.interim_results,
            punctuate=self.punctuate,
            diarize=self.diarize,
            timeout=self.timeout
        )
    
    async def aclose(self) -> None:
        """Close the STT and clean up resources."""
        pass
    
    def prewarm(self) -> None:
        """Pre-warm connection to Soniox service."""
        logger.info("Soniox STT prewarmed")


class SonioxRecognizeStream(RecognizeStream):
    """
    Real-time streaming speech recognition session for Soniox.
    
    Provides ultra-low latency transcription with WebSocket connection
    to Soniox's real-time API.
    """
    
    def __init__(
        self,
        *,
        api_key: str,
        model: str = "stt-rt-preview",
        language: str = "auto",
        sample_rate: int = 16000,
        interim_results: bool = True,
        punctuate: bool = True,
        diarize: bool = False,
        timeout: float = 30.0,
    ) -> None:
        """
        Initialize streaming session.
        
        Args:
            api_key: Soniox API key
            model: Soniox model to use
            language: Language code
            sample_rate: Audio sample rate
            interim_results: Whether to return interim results
            punctuate: Whether to add punctuation
            diarize: Whether to perform speaker diarization
            timeout: WebSocket timeout
        """
        dummy_stt = SonioxSTT(api_key=api_key, model=model, language=language)
        
        super().__init__(
            stt=dummy_stt,
            conn_options=APIConnectOptions(timeout=timeout),
            sample_rate=sample_rate
        )
        
        self.api_key = api_key
        self.model = model
        self.language = language
        self.sample_rate = sample_rate
        self.interim_results = interim_results
        self.punctuate = punctuate
        self.diarize = diarize
        self.timeout = timeout
        
        self._websocket = None
        self._listen_task = None
        self._final_tokens = []
        self._non_final_tokens = []
    
    async def _run(self) -> None:
        """Main run loop that processes audio input and manages WebSocket connection."""
        logger.info("Starting STT stream processing...")
        
        await self._connect()
        
        logger.info("Creating listening task...")
        self._listen_task = asyncio.create_task(self._listen())
        logger.info(f"Started listening task: {self._listen_task}")
        
        await asyncio.sleep(0.1)
        logger.info(f"Listening task done: {self._listen_task.done()}")
        if self._listen_task.done():
            logger.error(f"Listening task failed: {self._listen_task.exception()}")
        
        try:
            logger.info("Starting to process audio input...")
            async for item in self._input_ch:
                if isinstance(item, self._FlushSentinel):
                    logger.info("Received flush sentinel, sending empty data to Soniox")
                    if self._websocket:
                        try:
                            await self._websocket.send("")
                            logger.info("Sent empty data to flush")
                        except Exception as e:
                            logger.error(f"Error sending flush: {e}")
                else:
                    logger.debug(f"Processing audio frame: {type(item)}")
                    try:
                        await self.write(item)
                    except ConnectionClosed:
                        logger.warning("WebSocket connection lost, attempting to reconnect...")
                        self._websocket = None
                        await self._connect()
                        if self._websocket:
                            await self.write(item)
        except Exception as e:
            logger.error(f"Error in STT stream processing: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise
        finally:
            logger.info("Cleaning up STT stream...")
            if self._listen_task:
                self._listen_task.cancel()
                try:
                    await self._listen_task
                except asyncio.CancelledError:
                    pass
    
    async def _connect(self) -> None:
        """Establish WebSocket connection to Soniox streaming API."""
        if self._websocket is not None:
            return
        
        logger.info(f"Connecting to Soniox WebSocket API...")
        logger.info(f"URL: {SonioxSTT.WEBSOCKET_URL}")
        logger.info(f"Model: {self.model}")
        logger.info(f"Language: {self.language}")
        
        config = {
            "api_key": self.api_key,
            "model": self.model,
            "language_hints": [self.language] if self.language != "auto" else ["en"],
            "enable_language_identification": self.language == "auto",
            "enable_speaker_diarization": self.diarize,
            "enable_endpoint_detection": True,
            "audio_format": "pcm_s16le",
            "sample_rate": self.sample_rate,
            "num_channels": 1,
        }
        
        if not self.api_key or len(self.api_key) < 10:
            raise ValueError(f"Invalid API key: {self.api_key}")
        
        logger.info(f"API key length: {len(self.api_key)}")
        logger.info(f"API key prefix: {self.api_key[:10]}...")
        
        logger.info(f"Sending config to Soniox: {config}")
        
        try:
            self._websocket = await websockets.connect(
                SonioxSTT.WEBSOCKET_URL,
                ping_interval=20, 
                ping_timeout=10,
                close_timeout=10
            )
            
            logger.info("WebSocket connection established")
            
            config_json = json.dumps(config)
            logger.info(f"Sending config JSON: {config_json}")
            await self._websocket.send(config_json)
            
            await asyncio.sleep(0.1)
            
            logger.info("Checking for initial response from Soniox...")
            try:
                initial_response = await asyncio.wait_for(self._websocket.recv(), timeout=2.0)
                logger.info(f"Initial response from Soniox: {initial_response}")
            except asyncio.TimeoutError:
                logger.warning("No initial response from Soniox after 2 seconds")
            except Exception as e:
                logger.error(f"Error receiving initial response: {e}")
            
            logger.info("Connected to Soniox WebSocket streaming API")
            
            try:
                pong_waiter = await self._websocket.ping()
                await pong_waiter
                logger.info("WebSocket ping successful - connection is working")
            except Exception as e:
                logger.warning(f"WebSocket ping failed: {e}")
            
        except Exception as e:
            logger.error(f"Failed to connect to Soniox WebSocket API: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise
    
    async def _listen(self) -> None:
        """Listen for messages from Soniox WebSocket."""
        if not self._websocket:
            logger.warning("No WebSocket connection available for listening")
            return
        
        logger.info("Starting to listen for messages from Soniox...")
        logger.info(f"WebSocket state: {self._websocket.state}")
        
        try:
            logger.info("Waiting for messages from Soniox WebSocket...")
            logger.info(f"WebSocket state: {self._websocket.state}")
            
            logger.info("Starting async iteration over WebSocket messages...")
            
            last_message_time = asyncio.get_event_loop().time()
            
            async for message in self._websocket:
                last_message_time = asyncio.get_event_loop().time()
                logger.info(f"Received message from Soniox: {message[:200]}...")
                
                current_time = asyncio.get_event_loop().time()
                if current_time - last_message_time > 5.0:
                    logger.warning(f"No messages received for {current_time - last_message_time:.1f} seconds")
                
                try:
                    data = json.loads(message)
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse JSON: {e}")
                    continue
                
                if "error_code" in data:
                    logger.error(f"Soniox error: {data['error_code']} - {data['error_message']}")
                    break
                
                if data.get("finished"):
                    logger.info("Soniox session finished")
                    break
                
                tokens = data.get("tokens", [])
                self._non_final_tokens = []
                
                for token in tokens:
                    if token.get("text"):
                        if token.get("is_final"):
                            self._final_tokens.append(token)
                            logger.info(f"Final token: {token['text']}")
                        else:
                            self._non_final_tokens.append(token)
                            logger.info(f"Non-final token: {token['text']}")
                
                # Emit speech events
                if self._final_tokens or self._non_final_tokens:
                    await self._emit_speech_event()
                    
        except Exception as e:
            logger.error(f"Error in Soniox WebSocket listener: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
        finally:
            logger.info("Soniox WebSocket listener task ending")
            if self._websocket:
                logger.info(f"WebSocket state at end: {self._websocket.state}")
    
    async def _emit_speech_event(self) -> None:
        """Emit speech event with current tokens."""
        # Combine final and non-final tokens
        all_tokens = self._final_tokens + self._non_final_tokens
        
        if not all_tokens:
            return
        
        text = " ".join(token.get("text", "") for token in all_tokens if token.get("text"))
        
        if not text:
            return
        
        is_final = all(token.get("is_final", True) for token in all_tokens)
        
        event_type = (
            SpeechEventType.FINAL_TRANSCRIPT if is_final 
            else SpeechEventType.INTERIM_TRANSCRIPT
        )
        
        event = SpeechEvent(
            type=event_type,
            alternatives=[
                SpeechData(
                    language=self.language,
                    text=text
                )
            ]
        )
        
        await self._event_ch.send(event)
        
        logger.info(f"Soniox STT: {event.type.name} - '{text}'")
        
        if is_final:
            self._final_tokens = []
    
    async def write(self, frame: AudioFrame) -> None:
        """Write audio frame to the streaming session."""
        if self._websocket is None:
            logger.info("No WebSocket connection, connecting...")
            await self._connect()
        
        if self._websocket:
            try:
                audio_data = frame.data.tobytes()
                logger.debug(f"Sending {len(audio_data)} bytes of audio data")
                logger.debug(f"Audio frame info: shape={frame.data.shape}, sample_rate={frame.sample_rate}")
                await self._websocket.send(audio_data)
            except ConnectionClosed:
                logger.warning("WebSocket connection closed")
                raise
            except Exception as e:
                logger.error(f"Error sending audio frame: {e}")
                raise
    
    async def aclose(self) -> None:
        """Close the streaming session."""
        self._input_ch.close()
        
        if self._listen_task:
            self._listen_task.cancel()
            try:
                await self._listen_task
            except asyncio.CancelledError:
                pass
        
        if self._websocket:
            try:
                await self._websocket.send("")
            except:
                pass
            
            await self._websocket.close()
            self._websocket = None
        
        await super().aclose()


def create_soniox_stt(
    *,
    api_key: Optional[str] = None,
    model: str = "stt-rt-preview",
    language: str = "auto",
    sample_rate: int = 16000,
    interim_results: bool = True,
    punctuate: bool = True,
    diarize: bool = False,
    max_retries: int = 3,
    retry_delay: float = 1.0,
    timeout: float = 30.0,
) -> SonioxSTT:

    return SonioxSTT(
        api_key=api_key,
        model=model,
        language=language,
        sample_rate=sample_rate,
        interim_results=interim_results,
        punctuate=punctuate,
        diarize=diarize,
        max_retries=max_retries,
        retry_delay=retry_delay,
        timeout=timeout,
    )
