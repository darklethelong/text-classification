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
 * Chunk a conversation into groups of N utterances for analysis
 * @param {Array} utterances - Array of utterance objects
 * @param {number} chunkSize - Number of utterances per chunk
 * @returns {Array} Array of conversation chunks
 */
export const chunkConversation = (utterances, chunkSize = 4) => {
  const chunks = [];
  
  // Group utterances into chunks of size chunkSize
  for (let i = 0; i < utterances.length; i += chunkSize) {
    const chunk = utterances.slice(i, i + chunkSize);
    chunks.push({
      id: `chunk-${i / chunkSize + 1}`,
      utterances: chunk,
      isComplaint: false, // Will be filled in by analysis
      confidence: 0,
      text: chunk.map(u => `${u.speaker}: ${u.content}`).join('\n'),
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