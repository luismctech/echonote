/** Opaque identifier for a user-defined custom template. */
export type CustomTemplateId = string;

/** A user-defined summary prompt template. */
export interface CustomTemplate {
  /** Stable UUID identifier. */
  id: CustomTemplateId;
  /** Short display name (e.g. "Product Standup"). */
  name: string;
  /** System-prompt text sent to the LLM. */
  prompt: string;
}
