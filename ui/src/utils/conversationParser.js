/**
 * Parses a conversation string into structured format and chunks
 */

/**
 * Parse a conversation string into separate utterances
 * @param {string} text - The conversation text
 * @returns {Array} Array of utterance objects with speaker and text
 */
export const parseConversation = (text) => {
  // Split by new lines
  const lines = text.split('\n').filter(line => line.trim() !== '');
  const utterances = [];
  
  lines.forEach(line => {
    // Try to identify the speaker (Caller/Agent)
    const speakerMatch = line.match(/^(Caller|Agent|Customer|Representative|Rep|Support|User|System):\s*(.*)/i);
    
    if (speakerMatch) {
      let speaker = speakerMatch[1].toLowerCase();
      const content = speakerMatch[2].trim();
      
      // Normalize speaker names
      if (['customer', 'user'].includes(speaker)) {
        speaker = 'caller';
      } else if (['representative', 'rep', 'support', 'system'].includes(speaker)) {
        speaker = 'agent';
      }
      
      utterances.push({
        speaker,
        content,
        isComplaint: false, // Will be filled in later by analysis
      });
    } else {
      // If no speaker is identified, append to the previous utterance
      if (utterances.length > 0) {
        utterances[utterances.length - 1].content += ' ' + line.trim();
      } else {
        // If this is the first line and no speaker, assume it's the caller
        utterances.push({
          speaker: 'caller',
          content: line.trim(),
          isComplaint: false,
        });
      }
    }
  });
  
  return utterances;
};

/**
 * Chunk a conversation into groups of N utterances for analysis using a sliding window approach
 * @param {Array} utterances - Array of utterance objects
 * @param {number} chunkSize - Number of utterances per chunk
 * @returns {Array} Array of conversation chunks
 */
export const chunkConversation = (utterances, chunkSize = 4) => {
  const chunks = [];
  
  // If we have fewer utterances than the chunk size, just return one chunk
  if (utterances.length <= chunkSize) {
    return [{
      id: 'chunk-1',
      utterances: utterances,
      isComplaint: false,
      confidence: 0,
      text: utterances.map(u => `${u.speaker}: ${u.content}`).join('\n'),
      windowStart: 1,
      windowEnd: utterances.length
    }];
  }
  
  // Use sliding window approach to analyze overlapping chunks
  // For each possible starting position
  for (let i = 0; i <= utterances.length - chunkSize; i++) {
    const chunk = utterances.slice(i, i + chunkSize);
    chunks.push({
      id: `chunk-${i + 1}`,
      utterances: chunk,
      isComplaint: false, // Will be filled in by analysis
      confidence: 0,
      text: chunk.map(u => `${u.speaker}: ${u.content}`).join('\n'),
      windowStart: i + 1,
      windowEnd: i + chunkSize
    });
  }
  
  return chunks;
};

/**
 * Reconstructs the full conversation text from utterances
 * @param {Array} utterances - Array of utterance objects
 * @returns {string} Formatted conversation text
 */
export const reconstructConversation = (utterances) => {
  return utterances
    .map(u => `${u.speaker.charAt(0).toUpperCase() + u.speaker.slice(1)}: ${u.content}`)
    .join('\n');
};

/**
 * Calculate complaint percentage from analyzed chunks
 * @param {Array} chunks - Array of analyzed conversation chunks
 * @returns {number} Percentage of chunks classified as complaints
 */
export const calculateComplaintPercentage = (chunks) => {
  if (!chunks || chunks.length === 0) return 0;
  
  const complaintChunks = chunks.filter(chunk => chunk.isComplaint);
  return (complaintChunks.length / chunks.length) * 100;
};

/**
 * Calculate complaint percentage for caller utterances only
 * @param {Array} utterances - Array of utterance objects with isComplaint property
 * @returns {number} Percentage of caller utterances classified as complaints
 */
export const calculateCallerComplaintPercentage = (utterances) => {
  if (!utterances || utterances.length === 0) return 0;
  
  const callerUtterances = utterances.filter(u => u.speaker === 'caller');
  if (callerUtterances.length === 0) return 0;
  
  const complaintUtterances = callerUtterances.filter(u => u.isComplaint);
  return (complaintUtterances.length / callerUtterances.length) * 100;
};

/**
 * Reconstruct data from API response into frontend-friendly format.
 * This handles the sliding window data from the backend.
 * @param {Array} chunks - Array of chunks from API response
 * @param {Array} utterances - Array of all utterances
 * @returns {Object} Object with utterances and chunks
 */
export const reconstructFromApiResponse = (responseData) => {
  if (!responseData || !responseData.chunks || !responseData.chunks.length) {
    return {
      utterances: [],
      chunks: []
    };
  }
  
  // Extract all utterances in order
  const allUtterances = [];
  const seenUtteranceIds = new Map(); // Changed to Map to track order
  let utteranceOrder = 0;
  
  // Process all utterances from all chunks
  responseData.chunks.forEach(chunk => {
    chunk.utterances.forEach(utterance => {
      // Create a unique ID for each utterance based on content and speaker
      const utteranceId = `${utterance.speaker}:${utterance.content}`;
      
      if (!seenUtteranceIds.has(utteranceId)) {
        // Store the order of appearance in the Map
        seenUtteranceIds.set(utteranceId, utteranceOrder++);
        allUtterances.push({
          ...utterance,
          utteranceId, // Store the ID for sorting
          isComplaint: false // Default value, will be updated based on chunks
        });
      }
    });
  });
  
  // Sort utterances by their appearance order
  allUtterances.sort((a, b) => {
    return seenUtteranceIds.get(a.utteranceId) - seenUtteranceIds.get(b.utteranceId);
  });
  
  // Update isComplaint flag for utterances based on chunks
  responseData.chunks.forEach(chunk => {
    if (chunk.is_complaint) {
      chunk.utterances.forEach(chunkUtterance => {
        const matchingUtterance = allUtterances.find(u => 
          u.speaker === chunkUtterance.speaker && 
          u.content === chunkUtterance.content
        );
        
        if (matchingUtterance) {
          matchingUtterance.isComplaint = true;
        }
      });
    }
  });
  
  // Process chunks
  const processedChunks = responseData.chunks.map(chunk => {
    // Log to debug the window information
    console.log(`Processing chunk ID ${chunk.id} with window_start=${chunk.window_start}, window_end=${chunk.window_end}`);
    
    return {
      id: chunk.id,
      isComplaint: chunk.is_complaint,
      confidence: chunk.confidence,
      utterances: chunk.utterances.map(u => ({
        speaker: u.speaker,
        content: u.content,
        isComplaint: u.isComplaint || false
      })),
      // Ensure these properties are properly mapped
      windowStart: chunk.window_start || null,
      windowEnd: chunk.window_end || null,
      text: chunk.text
    };
  });
  
  console.log('Processed chunks with window information:', processedChunks);
  
  return {
    utterances: allUtterances,
    chunks: processedChunks
  };
}; 